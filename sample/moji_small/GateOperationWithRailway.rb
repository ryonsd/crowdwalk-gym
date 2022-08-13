# coding: utf-8
require 'csv'
require 'json'
require 'java'
require 'open3'
include Open3

import 'nodagumi.ananPJ.Scenario.CloseGateEvent'
import 'nodagumi.ananPJ.Scenario.OpenGateEvent'
import 'nodagumi.ananPJ.Scenario.AlertEvent'
import 'nodagumi.Itk.Itk'

#--======================================================================
#++
## CrowdWalk の EvacuationSimulator の制御のwrapper
class GateOperationWithRailway < CrowdWalkWrapper
  #--------------------------------------------------------------
  #++
  ## 初期化
  ## _simulator_:: java のシミュレータ(EvacuationSimulator)
  def initialize(simulator)
    super(simulator)
    @simulator = simulator
    @dummy_event = CloseGateEvent.new
    @step = 0
    @sim_step = 0
  end

  #--------------------------------------------------------------
  #++
  ## シミュレーション前処理
  def prepareForSimulation()
    # コンソールに状態表示をするか
    @monitor = $settings[:monitor]
    
    # ゲートノード(単一)
    @gate_node_tag = $settings[:gate_node_tag]
    nodes = []
    @networkMap.eachNode() {|node| nodes << node if node.matchTag("^#{@gate_node_tag}$") }
    unless nodes.size() == 1
      logError('GateOperation', "gate_node_tag error: #{@gate_node_tag}(#{nodes.size()} found)")
      exit(1)
    end
    @gate_node = nodes.first
    
    # ノードからリンクに入って来る人数をカウントする場合は true, 出て行く人数をカウントする場合は false
    @count_by_entering = $settings[:count_by_entering]
    
    # エージェントをカウントするノード(複数)
    @counting_positions = []
    $settings[:counting_positions].each do |position|
      node_tag = position[:node_tag]
      node_term = ItkTerm.newTerm(position[:node_tag])
      nodes = []
      @networkMap.eachNode() {|node| nodes << node if node.hasTag(node_term) }
      unless nodes.size() == 1
        logError('GateOperation', "node_tag error: #{node_tag}(#{nodes.size()} found)")
        exit(1)
      end
      node = nodes.first
      
      link_tag = position[:link_tag]
      link_term = ItkTerm.newTerm(position[:link_tag])
      links = []
      @networkMap.eachLink() {|link| links << link if link.hasTag(link_term) }
      unless links.size() == 1
        logError('GateOperation', "link_tag error: #{link_tag}(#{links.size()} found)")
        exit(1)
      end
      link = links.first
      link.enablePassCounter(node)
      
      @counting_positions << {node_tag: node_tag, node: node, link_tag: link_tag, link: link}
    end
    if @counting_positions.empty?
      logError('GateOperation', 'counting_positions is empty')
      exit(1)
    end
    
    # 乗車開始からゲートを開くまでの時間(s)
    delay_time = $settings[:delay_time]
    if delay_time == nil
      logError('GateOperation', 'delay_time error')
      exit(1)
    end
    @delay_time = delay_time.to_i
    
    # 乗車時刻, 収容人数, コメント
    @diagram = []
    CSV.read($settings[:diagram_file]).each do |diagram|
      @diagram << {time_str: diagram[0], time: time_to_int(diagram[0]), capacity: diagram[1].to_i, message: diagram[2]}
    end
    
    # 乗車エージェント数
    @count_of_passenger = 0

    # 
    fl = []
    @networkMap.eachLink() {|link| fl << link if link.hasTag("start_link")}
    @fl1 = fl.first

    command = "python " + $settings[:path_to_gym] + "tools/initialize.py" +
    " " + $settings[:path_to_gym] +
    " " + $settings[:env] +
    " " + $settings[:path_to_agent_log] +
    " " + $settings[:path_to_crowdwalk_config_dir]

    sleep 3

    o, e, s = Open3.capture3(command) # output, error, status
  end

  #--------------------------------------------------------------
  #++
  ## シミュレーション後処理
  def finalizeSimulation()
    sim_previous_step = (@step) * $settings[:step_duration]
    sim_final_step = @sim_step

    command = "python " + $settings[:path_to_gym] + "tools/finalize.py" +
    " " + $settings[:path_to_gym] +
    " " + $settings[:env] +
    " " + @step.to_s + 
    " " + sim_previous_step.to_s +
    " " + sim_final_step.to_s +
    " " + $settings[:path_to_crowdwalk_config_dir] +
    " " + $settings[:path_to_agent_log] +
    " " + $settings[:n_obj]

    o, e, s = Open3.capture3(command) # output, error, status
  end

  #--------------------------------------------------------------
  #++
  ## update の先頭で呼び出される。
  ## _simTime_:: シミュレーション内相対時刻
  def preUpdate(simTime)
    absoluteTime = simTime.getAbsoluteTime().to_i
    relativeTime = simTime.getRelativeTime().to_i
    
    # 通過エージェント数のカウント
    count_of_passing = 0
    @counting_positions.each do |position|
      count_of_passing += position[:link].getPassCount(position[:node], @count_by_entering)
    end
    
    # ホーム上または駅構内に存在するエージェント数
    count_of_standby = count_of_passing - @count_of_passenger
    
    next_index = @diagram.index {|train| train[:time] > absoluteTime }
    case next_index
    when nil  # 最終電車
      current_train = @diagram.reverse.find {|train| train[:time] <= absoluteTime }
      return unless current_train   # データが異常
      lastTrainGateOperation(absoluteTime, current_train, count_of_standby)
    when 0    # 最初の電車
      next_train = @diagram[next_index]
      beforeArrivalGateOperation(absoluteTime, next_train, count_of_standby)
    else      # 通常
      current_train = @diagram[next_index - 1]
      next_train = @diagram[next_index]
      normalGateOperation(absoluteTime, current_train, next_train, count_of_standby)
    end

    #-------------------------------
    # action selection
    if (relativeTime-1) % $settings[:step_duration] == 0
      # p absoluteTime
      
      # set guide to simulation
      # is_step = false
      # while !is_step do
      #   history = File.open($settings[:path_to_agent_log]+"history.json") do |f|
      #     JSON.load(f)
      #   end
      #   begin
      #     if !history[@step.to_s]["action"].to_f.nan? 
      #       action = history[@step.to_s]["action"]
      #       is_step = true
      #     end
      #   rescue 
      #     p "transition is not added yet"
      #     @step -= 1
      #     get_state_reward(relativeTime-1)
      #     action = 0
      #     is_step = true
      #   end
      # end

      check_action_selected(relativeTime)
      history = File.open($settings[:path_to_agent_log]+"history.json") do |f|
        JSON.load(f)
      end
      action = history[@step.to_s]["action"]
      # print "do ", action, "\n"


      if action == 0    
        term_1 = ItkTerm.newTerm("guide_route1")
        term_2 = ItkTerm.newTerm("guide_route2")
        @fl1.addAlertMessage(term_2, simTime, false)
        @fl1.addAlertMessage(term_1, simTime, true)
      elsif action == 1
        term_1 = ItkTerm.newTerm("guide_route1")
        term_2 = ItkTerm.newTerm("guide_route2")
        @fl1.addAlertMessage(term_1, simTime, false)
        @fl1.addAlertMessage(term_2, simTime, true)
      end

    end

    @sim_step += 1


  end

  #--------------------------------------------------------------
  #++
  ## 
  def check_action_selected(simTime)
    command = "python " + $settings[:path_to_gym] + "tools/check_action_selected.py" +
    " " + @step.to_s +
    " " + $settings[:path_to_agent_log]

    o, e, s = Open3.capture3(command)
  end

  #--------------------------------------------------------------
  #++
  ## update の最後に呼び出される。
  ## _relTime_:: シミュレーション内相対時刻
  def postUpdate(simTime)
    absoluteTime = simTime.getAbsoluteTime().to_i
    relativeTime = simTime.getRelativeTime().to_i
    if relativeTime % $settings[:step_duration] == 0 and relativeTime >= $settings[:step_duration]
      # print [@step, "get_state_reward"]
      get_state_reward(relativeTime)
    end
  end

  #--------------------------------------------------------------
  #++
  ## get state and reward
  def get_state_reward(relativeTime)
    
    command = "python " + $settings[:path_to_gym] + "tools/get_state_reward.py" +
    " " + $settings[:path_to_gym] +
    " " + $settings[:env] +
    " " + @step.to_s +
    " " + $settings[:step_duration].to_s +
    " " + relativeTime.to_s +
    " " + $settings[:path_to_crowdwalk_config_dir] +
    " " + $settings[:path_to_agent_log] +
    " " + $settings[:n_obj]

    o, e, s = Open3.capture3(command) # output, error, status

    @step += 1
    # puts "step"
  end

  # 最初の電車が到着する前(乗車可能になる前)のゲート制御
  #   乗客はホームで待機する
  #   ホームが一杯になったら(最初の電車の収容人数以上がホームで待機)ゲートを閉じる
  def beforeArrivalGateOperation(absoluteTime, next_train, count_of_standby)
    # ホームが一杯になったらゲートを閉じる
    if count_of_standby >= next_train[:capacity] and not @gate_node.isGateClosed(nil, nil)
      puts "#{Itk.formatSecTime(absoluteTime)} Gate closed" if @monitor
      @gate_node.addTag(Itk.intern('GATE_CLOSED'))
      @gate_node.closeGate(@gate_node_tag, @dummy_event)
    end
  end

  # 通常のゲート制御
  #   電車が到着してから @delay_time 秒経過するまでは乗車せずにホームで待ち続ける
  #   電車が到着してから @delay_time 秒後にホームで待っていた乗客が一瞬で一斉に乗車を完了する
  #   収容人数に満たない間は次の電車が来るまで乗車を続ける
  #   収容人数に達した後はホームで待機する
  #   ホームが一杯になったら(次の電車の収容人数以上がホームで待機)ゲートを閉じる
  #   次の電車の到着1秒前に発進するものとする
  def normalGateOperation(absoluteTime, current_train, next_train, count_of_standby)
    next_capacity = next_train[:capacity]
    
    # 電車が到着して乗車可能になった(ただしまだ乗車しない)
    if absoluteTime == current_train[:time]
      capacity = current_train[:capacity]
      # 乗車開始
      puts "#{Itk.formatSecTime(absoluteTime)} Door opened: #{current_train[:message]}, standby: #{count_of_standby}, traincapa: #{capacity}" if @monitor
    
    # 乗車可能な状態(ただしまだ乗車しない)
    elsif absoluteTime < (current_train[:time] + @delay_time)
      capacity = current_train[:capacity]
    
    # ホームで待機していた乗客が(一瞬で)全員乗車した
    elsif absoluteTime == (current_train[:time] + @delay_time)
      if current_train[:capacity] > count_of_standby
        current_train[:capacity] -= count_of_standby
      else
        # 電車の収容人数を超えた分も全員乗車出来たものとする
        current_train[:capacity] = 0
      end
      @count_of_passenger += count_of_standby
      count_of_standby = 0
      capacity = current_train[:capacity] + next_capacity
      
      # (乗客が全員乗車してホームが空いたため)閉じているゲートを開く
      if @gate_node.isGateClosed(nil, nil)
        puts "#{Itk.formatSecTime(absoluteTime)} Gate opened" if @monitor
        @gate_node.removeTag(Itk.intern('GATE_CLOSED'))
        @gate_node.openGate(@gate_node_tag, @dummy_event)
      end
    
    # この電車が満員でなければ乗車可能、満員ならば次の電車待ち
    else
      if current_train[:capacity] > 0
        if current_train[:capacity] > count_of_standby
          current_train[:capacity] -= count_of_standby
          @count_of_passenger += count_of_standby
          count_of_standby = 0
        else
          @count_of_passenger += current_train[:capacity]
          count_of_standby -= current_train[:capacity]
          current_train[:capacity] = 0
        end
      end
      capacity = current_train[:capacity] + next_capacity
    end
    
    # ホームが一杯になったらゲートを閉じる
    if count_of_standby >= capacity and not @gate_node.isGateClosed(nil, nil)
      puts "#{Itk.formatSecTime(absoluteTime)} Gate closed" if @monitor
      @gate_node.addTag(Itk.intern('GATE_CLOSED'))
      @gate_node.closeGate(@gate_node_tag, @dummy_event)
    end
  end

  # 最終電車が到着した後のゲート制御
  #   電車が到着してから @delay_time 秒経過するまでは乗車せずにホームで待ち続ける
  #   電車が到着してから @delay_time 秒後にホームで待っていた乗客が一瞬で一斉に乗車を完了する
  #   収容人数に満たない間は乗車を続ける
  #   収容人数に達したらゲートを閉じる
  #   収容人数に達するまで電車は発進しないものとする
  def lastTrainGateOperation(absoluteTime, current_train, count_of_standby)
    # 電車が到着して乗車可能になった(ただしまだ乗車しない)
    if absoluteTime == current_train[:time]
      # 乗車開始
      puts "#{Itk.formatSecTime(absoluteTime)} Door opened: #{current_train[:message]}" if @monitor
      puts "#{Itk.formatSecTime(absoluteTime)} This is the last train." if @monitor
    
    # 乗車可能な状態(ただしまだ乗車しない)
    elsif absoluteTime < (current_train[:time] + @delay_time)
      # no operation
    
    # ホームで待機していた乗客が(一瞬で)全員乗車した
    elsif absoluteTime == (current_train[:time] + @delay_time)
      if current_train[:capacity] > count_of_standby
        current_train[:capacity] -= count_of_standby
      else
        # 電車の収容人数を超えた分も全員乗車出来たものとする
        current_train[:capacity] = 0
      end
      @count_of_passenger += count_of_standby
      count_of_standby = 0
    
    # この電車が満員でなければ乗車可能
    else
      if current_train[:capacity] > 0
        if current_train[:capacity] > count_of_standby
          current_train[:capacity] -= count_of_standby
          @count_of_passenger += count_of_standby
          count_of_standby = 0
        else
          # 電車の収容人数を超えた分も全員乗車出来たものとする
          @count_of_passenger += count_of_standby
          current_train[:capacity] = 0
          count_of_standby = 0
        end
      elsif count_of_standby > 0
        # 収容人数をオーバーしても駅構内にいるエージェントは全員乗車させる
        @count_of_passenger += count_of_standby
        count_of_standby = 0
      end
    end
    
    # 電車が満員ならばゲートを閉じる
    if count_of_standby >= current_train[:capacity] and not @gate_node.isGateClosed(nil, nil)
      puts "#{Itk.formatSecTime(absoluteTime)} Gate closed" if @monitor
      @gate_node.addTag(Itk.intern('GATE_CLOSED'))
      @gate_node.closeGate(@gate_node_tag, @dummy_event)
    end
  end

  def time_to_int(time_str)
    return 0 unless time_str =~ /(\d{1,2}):(\d{1,2}):(\d{1,2})/
    $1.to_i * 3600 + $2.to_i * 60 + $3.to_i
  end
end
