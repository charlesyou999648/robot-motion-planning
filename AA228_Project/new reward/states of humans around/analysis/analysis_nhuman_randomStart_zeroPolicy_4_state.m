%% world
clear;
clc;
% load data should have nHumans value
load('test_4.mat') % good

%% test

collisionRate = [];
reachGoalRate = [];
collisionNumberRate = [];
meanTravellingTime = [];

for nHumans = 1:20
    totalTests = 100;

    totalIter = 1;
    iters = [];
    scores = [];
    finals = [];
    numberCollisions = [];
    collisions = zeros(totalTests, 1);
    reachGoal = zeros(totalTests, 1);
    while totalIter <= totalTests

        %%% human motion setup

        % random start and end positions for humans. They must go to the other side
        % of the hall way.
        humanStart=[randi([10,endpoint-10],[nHumans,1]),-halfWidth+2*halfWidth*randi([0,1],[nHumans,1])];%(3, 2)
        humanEnd=[randi([10,endpoint-10],[nHumans,1]),-humanStart(:,2)];%(3, 2)
        humanVel = [];
        humanPos = humanStart; %(3, 2)
        for i = 1:nHumans
            humanVelDir = humanEnd(i,:)-humanStart(i,:);
            humanVel = [humanVel; humanVelDir/norm(humanVelDir) * humanVelMag(1)]; % at first the magnitude is the largest
        end

        humanStartTime = randi([earliestStartTime,latestStartTime],nHumans,1);

        %%% robot motion set up
        robotPos = [0,0];
        robotVel = [0,0];

        % intersections between robot trajectory and human trajectories
        crossPoint = (humanStart+humanEnd)/2;%(3, 2)

        % initial state and actions
        s = zeros(4,1);
        s(3) = 1; % initially it is always possible to cause collision (crosspoint)
        atPlus = randi([3,5]);% intial action should be zero or positive action

        % iteration set up
        % Here the iteration is for robot simulation in one run
        iter = 0;
        humanPosH = zeros(2,iterLimit,nHumans); % human position history
        humanPosH(:,1,:) = humanPos'; % the first timestamp
        robotPosH = robotPos; % robot position history
        robotVelH = robotVel(1); % only x component of robot velocity is meaningful
        actionH = []; % action history
        score = 0;
        
        HumansCollided = [];

        while iter < iterLimit

            iter = iter + 1;
            st = s;
            at = atPlus;

            % update robot position
            robotPosPast = robotPos;
            robotPos = robotPos + dt * robotVel + [1/2 * action(at) * dt^2, 0];
            robotVel = robotVel + [action(at) * dt, 0];

            % update human position

            for i = 1:nHumans
                if iter >= humanStartTime(i)
                    humanPos(i, :) = humanPos(i, :)+ dt * humanVel(i, :);
                end
                if abs(humanPos(i,2)) > abs(humanEnd(i, 2)) % not between -200, 200
                    humanVel(i, :) = [0,0]; % human stopped after reaching the destination
                end
            end

            % robot position hitory, human position history,
            % robot velocity history and action history updated
            robotPosH = [robotPosH; robotPos];
            humanPosH(:,iter,:) = humanPos';
            robotVelH = [robotVelH, robotVel(1)];
            actionH = [actionH, at];


            %humanPos (3, 2)
            distRobotHuman = [];
             for i = 1:nHumans
                distRobotHumanI = norm(humanPos(i, :)-robotPos);
                distRobotHuman = [distRobotHuman, distRobotHumanI];
                if distRobotHumanI > rHuman(2)
                    humanVelMagI = humanVelMag(1); % far then fast
                elseif distRobotHumanI > rHuman(1)
                    humanVelMagI = humanVelMag(2); % close then slow
                else
                    humanVelMagI = humanVelMag(3);
                end

                if norm(humanVel(i,:)) ~= 0
                     humanVel(i, :) = humanVel(i, :) / norm(humanVel(i,:)) * humanVelMagI;
                end    
            end
            % find the closest human and the distance between that human and
            % the robot
            [closeDistRobotHuman, closeIdx] = min(distRobotHuman);


            % state1 - 0: not reach the goal, 1: reach the goal
            % state2 - 0: r > rRobot(3), 1: rRobot(3) > r > rRobot(2), 2: rRobot(2) > r > rRobot(1), 3: robot(1) > r
            % state3 - 0: velocity directions won't cause collision, 1: will cause collision
            % state4 - 0: velocity is inside nominalRobotVelRange(1), 
            %          1: velocity is outside nominalRobotVelRange(1) but inside nominalRobotVelRange(2),
            %          2: velocity is outside nominalRobotVelRange(2)
            % state update
            if robotPos(1) >= endpoint
                s(1) = 1;
            else
                s(1) = 0;
            end

            if closeDistRobotHuman > rRobot(3)
                s(2) = 0;
            elseif (closeDistRobotHuman < rRobot(3) && closeDistRobotHuman > rRobot(2))
                s(2) = 1;
            elseif (closeDistRobotHuman < rRobot(2) && closeDistRobotHuman > rRobot(1))
                s(2) = 2;
            elseif closeDistRobotHuman < rRobot(1)
                s(2) = 3; % collision
                collisions(totalIter) = 1;
                if ~any(HumansCollided==closeIdx)
                    HumansCollided = [HumansCollided, closeIdx];
                end
            end

            pathVecRobot = crossPoint(closeIdx,:) - robotPos;
            pathVecHuman = crossPoint(closeIdx,:) - humanPos(closeIdx,:);

            if dot(pathVecRobot, robotVel) < 0 || dot(pathVecHuman, humanVel(closeIdx,:)) < 0
                s(3) = 0;
            else
                s(3) = 1;
            end

            robotVelDev = abs(robotVel(1) - nominalRobotVel);
            if robotVelDev < nominalRobotVelRange(1)
                s(4) = 0;
            elseif robotVelDev < nominalRobotVelRange(2)
                s(4) = 1;
            else
                s(4) = 2;
            end



            % epsilon-greedy
            if eDecay % is eDecay == True
                epsilon = exp(-eDecayFactor/totalEpisodes * totalIter);
            else
                epsilon = eConst; 
            end


            [tmp, atPlus] = max(Q(s(1)+1, s(2)+1, s(3)+1, s(4)+1, :));% we need index

            score = reward(s, at, w) + gamma * score;

            if robotPosPast(1) > endpoint
                reachGoal(totalIter) = 1;
                break;
            end  
        end
        iters = [iters, iter];
        scores = [scores, score];
        finals = [finals, robotPos(1)];
        numberCollisions = [numberCollisions, length(HumansCollided)];
        if mod(totalIter, 100) == 0
            fprintf('%.0f tests have beed done.\n', totalIter)
        end
        totalIter = totalIter + 1;
    end
    
    collisionRate = [collisionRate, mean(collisions)];
    reachGoalRate = [reachGoalRate, mean(reachGoal)];
    collisionNumberRate = [collisionNumberRate, mean(numberCollisions)];
    meanTravellingTime = [meanTravellingTime, mean(iters)*dt];

    fprintf('Number of humans: %.0f\n', nHumans)
    fprintf('Ratio of robots that collide during the test: %.3f\n', mean(collisions))
    fprintf('Ratio of robots that reach the goal during the test: %.3f\n', mean(reachGoal))
    fprintf('Ratio of number of humans collide with robots during the test: %.3f\n', mean(numberCollisions))
    fprintf('mean travelling time: %.3f\n\n',mean(iters)*dt)

end

%% analysis

nHumansList = 1:20;
figure;
plot(nHumansList, collisionRate,'*');
title('collisionRate')
figure;
plot(nHumansList, reachGoalRate,'o');
title('reachGoalRate')
figure;
plot(nHumansList, collisionNumberRate,'^');
title('collisionNumberRate')
figure;
plot(nHumansList, meanTravellingTime,'^');
title('meanTravellingTime')

% figure;
% hold on;
% for i = 1:totalTests
%     if collisions(i)==1
%         plot(i, scores(i),'r*')
%     else
%         plot(i, scores(i),'b*')
%     end
% end
% % plot(scores,'*')
% figure;
% hold on;
% for i = 1:totalTests
%     if reachGoal(i)==1
%         plot(i, scores(i),'go')
%     else
%         plot(i, scores(i),'ro')
%     end
% end
% fprintf('Ratio of robots that do not collide during the test: %.3f\n', 1-mean(collisions))
% fprintf('Ratio of robots that reach the goal during the test: %.3f\n', mean(reachGoal))




%% demo
% 
% % robotPosH -> (226, 2)
% % humanPosH -> (2, 100000, 3)
% 
% 
% 
% 
% robotPosH = robotPosH(1:size(robotPosH, 1)-1, :);
% humanPosHValid = humanPosH(:,1:size(robotPosH, 1),:);%(2, 225, 3)
% 
% robotPos = robotPosH;
% robotPos = [robotPos, zeros(size(robotPos, 1),1)];
% 
% 
% humanPos = {}
% 
% for i = 1:nHumans
%     humanPosTmp = humanPosHValid(:,:,i);
%     humanPosTmp = humanPosTmp';
%     humanPosTmp = [humanPosTmp, zeros(size(humanPosTmp, 1),1)];
%     humanPos{end+1} = humanPosTmp;
% end
% 
% close all
% clf('reset');
% hold on;
% 
% subplot(2,1,2)
% % axis equal
% set(gca,'position',[0.05,0.05,0.9,0.9])
% endpoint = 800; % 1400
% height = 300;
% hallwayWeight = 200;
% radicar = 80;
% 
% % draw pathway
% x_dist = 0:endpoint;
% plot(x_dist, -hallwayWeight + 0.*x_dist, 'Color','k','LineWidth',4)
% hold on
% plot(x_dist, 0 + 0.*x_dist, 'w--','Color','g','LineWidth',2)       
% plot(x_dist, hallwayWeight + 0.*x_dist, 'Color','k','LineWidth',4)
% axis([0,endpoint,-height,height]);
% plot(endpoint, 0, '*', 'Color','r','MarkerSize',20)
% 
% 
% % define human object
% gHuman={};
% for i = 1:nHumans
%     start1 = 400;
%     end1 = 400;
%     h1 = rectangle('position', [-10, -10, 20, 20], 'FaceColor', 'b');
%     g1 = hgtransform;
%     set(h1,'Parent',g1)
%     gHuman{end+1} = g1;
%     startpoint1 = [start1 hallwayWeight 0];
%     endpoint1 = [end1 -hallwayWeight 0];
%     
% end
% 
% % define vehicle
% robotCirclePos = [-rRobot(1), -rRobot(1), 2 * rRobot(1), 2 * rRobot(1)];
% robotFilledCircle = rectangle('Position',robotCirclePos,'Curvature',[1 1], 'FaceColor', 'red', 'Edgecolor','none');
% g = hgtransform;
% set(robotFilledCircle,'Parent',g)
% 
% 
% pt1 = [0 0 0];
% pt2 = [endpoint 0 0];
% 
% 
% % define circle
% % hcircle = viscircles([0,0],rRobot(1));
% gcircle = hgtransform;
% hcircle1 = viscircles([0,0],rRobot(2));
% hcircle2 = viscircles([0,0],rRobot(3));
% % set(hcircle,'Parent',gcircle)
% set(hcircle1,'Parent',gcircle)
% set(hcircle2,'Parent',gcircle)
% % moving
% for t=1:size(robotPos, 1)
%   g.Matrix = makehgtform('translate',robotPos(t,:));
%   gcircle.Matrix = g.Matrix;
%   for i = 1:nHumans
%       gHuman{i}.Matrix = makehgtform('translate',humanPos{i}(t,:));
%   end
%   drawnow
%   pause(0.01)
% end
% fprintf('score: %.2f\n', scores(end))
