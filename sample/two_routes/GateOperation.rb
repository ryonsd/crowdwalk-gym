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
## 
class GateOperation < CrowdWalkWrapper
  #--------------------------------------------------------------
  #++
  ## 
  def initialize(simulator)
    super(simulator)
    @simulator = simulator
    @dummy_event = CloseGateEvent.new
    @step = 0
  end

  #--------------------------------------------------------------
  #++
  ## preprocessing of simulation
  def prepareForSimulation()
    # コンソールに状態表示をするか
    @monitor = $settings[:monitor]

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

    fl = []
    @networkMap.eachLink() {|link| fl << link if link.hasTag("start_link")}
    @fl1 = fl.first

  end

  #--------------------------------------------------------------
  #++
  ## postprocessing of simulation
  def finalizeSimulation()
  end

  #--------------------------------------------------------------
  #++
  ## update の先頭で呼び出される
  ## _simTime_:: シミュレーション内相対時刻
  def preUpdate(simTime)
    absoluteTime = simTime.getAbsoluteTime().to_i
    
    # action selection
    if (absoluteTime-1) % $settings[:step_duration] == 0 and absoluteTime >= $settings[:step_duration]
      # p absoluteTime

      # get state and reward
      command = "python " + $settings[:path_to_gym_envs] + "get_state_reward.py " + $settings[:env] + " " + $settings[:step_duration].to_s +  " " + (absoluteTime-1).to_s  + " " + Dir.getwd

      o, e, s = Open3.capture3(command) # output, error, status
      history = File.open("log/history.json") do |f|
        JSON.load(f)
      end

      state = history["state"]
      reward = history["reward"]

      # select action and set guide to simulation
      action = action_selection(@step)

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

      @step += 1
    end

    # 通過エージェント数のカウント
    count_of_position = 0
    count_of_passing = 0
    count_of_standby = 0

  end


  def action_selection(step)
    


    if step % 2 == 0
      action = 0
    else
      action = 1
    end 

    return action
  end

  #--------------------------------------------------------------
  #++
  ## update の最後に呼び出される。
  ## _relTime_:: シミュレーション内相対時刻
  def postUpdate(simTime)
  end

  def time_to_int(time_str)
    return 0 unless time_str =~ /(\d{1,2}):(\d{1,2}):(\d{1,2})/
    $1.to_i * 3600 + $2.to_i * 60 + $3.to_i
  end
end
