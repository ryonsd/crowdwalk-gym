#! /usr/bin/env ruby
## -*- mode: ruby -*-
## = Sample Agent for CrowdWalk
## Author:: Itsuki Noda
## Version:: 0.0 2015/06/28 I.Noda
## Version:: 1.0 2018/10/31 R.Nishida [change calcCost]
##
## === History
## * [2014/06/28]: Create This File.
## * [YYYY/MM/DD]: add more
## == Usage
## * ...

require "date" ;
require 'RubyAgentBase.rb' ;
require 'GateOperation' ;

#--======================================================================
#++
## SampleAgent class
class UtilityAgent < RubyAgentBase
	
	#--============================================================
	#--::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	#++
	## Java から Ruby を呼び出すTriggerでのFilter。
	## この配列に Java のメソッド名（キーワード）が入っていると、
	## Ruby 側が呼び出される。入っていないと、無視される。
	## RubyAgentBase を継承するクラスは、このFilterを持つことが望ましい。
	## このFilterは、クラスをさかのぼってチェックされる。
	TriggerFilter = [
	#                   "preUpdate",
	#                   "update",
					"calcCostFromNodeViaLink",
	                "calcSpeed",
	                "calcAccel",
					"setGoal",
					"thinkCycle",
					] ;

	#--@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	#++
	## call counter
	attr_accessor :counter ;
	
	#--------------------------------------------------------------
	#++
	## シミュレーション各サイクルの前半に呼ばれる。
	def initialize(agent, config, fallback) 
			@counter = 0 ;
			@@finish = 0;
			@utilityBranch1Route1 = 0
			@utilityBranch1Route2 = 0
			@P = 1.0
			
			@b_distance = -18.260
			@b_guide = 6.346
			# @b_guide = 100

		super ;
	end

	#--------------------------------------------------------------
	#++
	## シミュレーション各サイクルの前半に呼ばれる。
	def preUpdate()
	#    p ['SampleAgent', :preUpdate, getAgentId(), currentTime()] ;
		@counter += 1;
		return super()
	end

	#--------------------------------------------------------------
	#++
	## シミュレーション各サイクルの後半に呼ばれる。
	def update()
	#    p ['SampleAgent', :update, getAgentId(), currentTime()] ;
		@counter += 1;
		return super() ;
	end
	
	def getSimulator()
		@counter += 1;
		return super();
	end
		
	#--------------------------------------------------------------
	#++
	## あるwayを選択した場合の目的地(_target)までのコスト。
	## _way_:: 現在進もうとしている道
	## _node_:: 現在の分岐点
	## _target_:: 最終目的地


	## message
	Term_guide_route1 = ItkTerm.ensureTerm("guide_route1");
	Term_guide_route2 = ItkTerm.ensureTerm("guide_route2");  

	def calcCostFromNodeViaLink(link, node, target)
		if getAgentId() then

			if (hasPlaceTag('start_link')) && ( link.getTags().contains('route1_1') || link.getTags().contains('route2_1') ) then
				# route1
				if link.getTags().contains('route1_1') then

					distance = 0 #0.39
					guide = 0;

					if(listenAlert(Term_guide_route1)) then
						guide = 1 ;
					else
						guide = 0 ;
					end

					@utilityBranch1Route1 = @b_distance*distance + @b_guide*guide
				end
				
				# route2
				if link.getTags().contains('route2_1') then

					distance = 0 #0.75
					guide = 0;

					if(listenAlert(Term_guide_route2)) then
						guide = 1 ;
					else
						guide = 0 ;
					end
			
					@utilityBranch1Route2 = @b_distance*distance + @b_guide*guide
				
				end

	
				@P = Math.exp(@utilityBranch1Route1) / (Math.exp(@utilityBranch1Route1) + Math.exp(@utilityBranch1Route2))
				# p @P
				if Random.rand(1.0) <= @P then
					# p "go to R1"
					if link.getTags().contains('route1_1') then
						cost = -1000.0
					else
						cost = 0.0
					end

				else
					# p "go to R2"
					if link.getTags().contains('route1_1') then
						cost = 0.0
					else
						cost = -1000.0
					end
				end

				return cost
			end 

		
		end


		@counter += 1;
		cost = super(link, node, target);
		return cost
  	end

	#--------------------------------------------------------------
	#++
	## 速度を計算する。
	## _previousSpeed_:: 前のサイクルの速度。
	## *return* 速度。
	def calcSpeed(previousSpeed)
		#    p ['SampleAgent', :calcSpeed, getAgentId(), currentTime()] ;
		@counter += 1;
		# self.javaAgent.setEmptySpeed(1.5)
		return super(previousSpeed) ;
	end

	#--------------------------------------------------------------
	#++
	## 加速度を計算する。
	## _baseSpeed_:: 自由速度。
	## _previousSpeed_:: 前のサイクルの速度。
	## *return* 加速度。
	def calcAccel(baseSpeed, previousSpeed)
	#    p ['SampleAgent', :calcAccel, getAgentId(), currentTime()] ;
		@counter += 1;
		return super(baseSpeed, previousSpeed) ;
	end

	#--------------------------------------------------------------
	#++
	## 思考ルーチン。
	## ThinkAgent のサンプルと同じ動作をさせている。
	def thinkCycle()
	#    p ['SampleAgent', :thinkCycle, getAgentId(), currentTime()] ;
		@counter += 1;
	#    return super ;
		return ItkTerm::NullTerm ;
	end

end # class SampleAgent

