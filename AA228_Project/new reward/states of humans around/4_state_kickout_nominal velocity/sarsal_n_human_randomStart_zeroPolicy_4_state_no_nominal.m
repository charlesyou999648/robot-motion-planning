%% world

clear;
clc;

% unit: cm, second

% key parameters

totalEpisodes = 100; % number of episodes in training process
iterLimit = 10000; % in each episode, the limit of number of actions a robot can take. i.e. one episode lasts at most 200 seconds.

eDecay = false;
eDecayFactor = 5; % => exp(-eDecayFactor/totalEpisodes * totalIter) = exp(-5/totalEpisodes * totalIter)
eConst = 0.1;

endpoint = 800; % length of hall way
halfWidth = 200; % half width of hall way
rRobot = [20, 60, 120]; % radii of circles centered at robot used to judge the distance between the human and the robot
nominalRobotVel = 50; % robot's nominal velocity is 50cm/s. Can change it into piecewise function the input of which is the traveled distance 
nominalRobotVelRange = [15, 30]; % ranges of velocities used to judge the deviation of the robot's real velocity from the nominal one 
humanVelMag= [50, 30, 10]; % initial human velocity. It will change based on the distance between human and robot.
rHuman = [60, 120]; % the distance that humans believe is safe or dangerous 
nHumans = 10; % number of humans walking in the hall way
earliestStartTime = 0;
latestStartTime = 30; % (x0.1 s)latest start time (by experience, it usually takes around 100~200 for the robot to finish the episode
nHumansAround = [2, 4];

%      1    2   3   4   5  6   7   8   9    10
w = [2000, -2, -2, -2, -1, 0, -2, -4, -1, -200];
reward = @(s, a, w) dot(w(1:3), [s(1), s(4)^2 *s(2)^3, s(4)^2 *s(3)*s(2)^3]) + w(3+a) + w(9) + w(10) * s(4)^2 * (s(2)==3); % a=3-> w(7)=0

% r = w1 * state1 + w2 * (state4 ^ 2 * state2 ^3) + w3 * (state4 ^ 2 * state3 * state2 ^3) + [w4, w5, w6, w7, w8](a) + w9 + w10 * state4 ^ 2 * (state2==3)
% reasoning for scale of w
% w1: positive reward for reaching the goal. Should be pretty large.
% w2: negative reward for being too close to humans. (s2 = 3 is collision)
% w3: negative reward for moving in the direction which may cause collision with humans (we usually just set w3 = w2)
% w4, w5, w6, w7, w8: negative reward for driving the robot. Acceleration cost always higher than deceleration at the same scale.
% w9: negative reward for time cost
% w10: extra negative reward for collision



%% state space

% state1 - 0: not reach the goal, 1: reach the goal
% state2 - 0: r > rRobot(3), 1: rRobot(3) > r > rRobot(2), 2: rRobot(2) > r > rRobot(1), 3: robot(1) > r
% state3 - 0: velocity directions won't cause collision, 1: will cause collision
% state4 - 1: < nHumansAround(1) humans inside rRobot(3)
%          2: > nHumansAround(1) and < nHumansAround(2) humans inside rRobot(3)
%          3: > nHumansAround(2) humans inside rRobot(3)

% for multiple humans, right now we want to just check the closest one
% state2 = 3 is literally the collision

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this can punish both negative velocity and the too fast positive speed
% we can try to decay the nominal speed when the robot is approaching the goal. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% s = zeros(4,1);


%% action space
action = [-20, -10, 0, 10, 20];
% hard deceleration, mild deceleration, zero acceleration, 
% mild acceleration, hard acceleration

% a = 3; % intial action is maintaining the speed. a can be 1,2,3,4,5

%% reward function

% state1 - 0: not reach the goal, 1: reach the goal
% state2 - 0: r > rRobot(3), 1: rRobot(3) > r > rRobot(2), 2: rRobot(2) > r > rRobot(1), 3: robot(1) > r
% state3 - 0: velocity directions won't cause collision, 1: will cause collision
% state4 - 1: < nHumansAround(1) humans inside rRobot(3)
%          2: > nHumansAround(1) and < nHumansAround(2) humans inside rRobot(3)
%          3: > nHumansAround(2) humans inside rRobot(3)

% r = w1 * state1 + w2 * (state5 ^ 2 * state2 ^3) + w3 * (state5 ^ 2 * state3 * state2 ^3) + [w4, w5, w6, w7, w8](a) + w9 + w10 * state5 ^ 2 * (state2==3)

% reasoning for scale of w
% w1: positive reward for reaching the goal. Should be pretty large.
% w2: negative reward for being too close to humans. (s2 = 3 is collision)
% w3: negative reward for moving in the direction which may cause collision with humans (we usually just set w3 = w2)
% w4, w5, w6, w7, w8: negative reward for driving the robot. Acceleration cost always higher than deceleration at the same scale.
% w9: negative reward for time cost
% w10: extra negative reward for collision




%% Simulator


% state1 - 0: not reach the goal, 1: reach the goal
% state2 - 0: r > rRobot(3), 1: rRobot(3) > r > rRobot(2), 2: rRobot(2) > r > rRobot(1), 3: robot(1) > r
% state3 - 0: velocity directions won't cause collision, 1: will cause collision
% state4 - 1: < nHumansAround(1) humans inside rRobot(3)
%          2: > nHumansAround(1) and < nHumansAround(2) humans inside rRobot(3)
%          3: > nHumansAround(2) humans inside rRobot(3)

% parameters
alpha = 0.2; % learning rate
gamma = 0.99; % discount factor
lambda=0.9; % lambda in sweeping updates

Q = zeros(2, 4, 2, 3, 5); %240
N = zeros(2, 4, 2, 3, 5);
stateCount = zeros(2, 4, 2, 3, 5);
dt = 0.1;
totalIter = 1;
iters = [];
scores = [];
finals = [];
collisions = zeros(totalEpisodes, 1);
reachGoal = zeros(totalEpisodes, 1);

%% train
while totalIter <= totalEpisodes

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
    s(4) = 1; % initially it is 1
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
        end

        pathVecRobot = crossPoint(closeIdx,:) - robotPos;
        pathVecHuman = crossPoint(closeIdx,:) - humanPos(closeIdx,:);
     
        if dot(pathVecRobot, robotVel) < 0 || dot(pathVecHuman, humanVel(closeIdx,:)) < 0
            s(3) = 0;
        else
            s(3) = 1;
        end
        
        nHumansAroundReal = 0;
        
        for i = 1:nHumans
            if distRobotHuman(i) < rRobot(3)
                nHumansAroundReal = nHumansAroundReal + 1;
            end
        end
        
        if nHumansAroundReal < nHumansAround(1)
            s(4) = 1;
        elseif nHumansAroundReal < nHumansAround(2)
            s(4) = 2;
        else
            s(4) = 3; 
        end
        

        % epsilon-greedy
        if eDecay % is eDecay == True
            epsilon = exp(-eDecayFactor/totalEpisodes * totalIter);
        else
            epsilon = eConst; 
        end
        
        % SARSA Lambda
        
        if rand() < epsilon % random choice
            atPlus = randi([1,5]);
        else % best choice
            [tmp, atPlus] = max(Q(s(1)+1, s(2)+1, s(3)+1, s(4), :));% we need index
        end
        
        tPlusIdx = sub2ind(size(Q), s(1)+1, s(2)+1, s(3)+1, s(4), atPlus);
        QtPlus = Q(tPlusIdx);
        
        tIdx = sub2ind(size(Q), st(1)+1, st(2)+1, st(3)+1, st(4), at);
        Qt = Q(tIdx);
        
        Nt = N(tIdx);
        N(tIdx) = Nt + 1;
        delta = reward(st, at, w) + gamma * QtPlus - Qt;
        stateCount(tIdx) = stateCount(tIdx) + 1;

        % stateCount = zeros(2, 4, 2, 3, 3, 5);
        for s1 = 1:2
            for s2 = 1:4
                for s3=1:2
                    for s4=1:3
                        for aa=1:5
                            tmpIdx = sub2ind(size(Q), s1,s2,s3,s4,aa);
                            Q(tmpIdx) = Q(tmpIdx) + alpha * delta * N(tmpIdx);
                            N(tmpIdx) = gamma * lambda * N(tmpIdx);
                        end
                    end
                end
            end
        end      

%         Q(tIdx) = Qt + alpha * (reward(st, at, w) + gamma * QtPlus - Qt); % vanilla SARSA
        
        score = reward(s, at, w) + gamma * score;
        
        if robotPosPast(1) > endpoint
            reachGoal(totalIter) = 1;
            break;
        end  
    end
    iters = [iters, iter];
    scores = [scores, score];
    finals = [finals, robotPos(1)];
    disp(totalIter);
    totalIter = totalIter + 1;
end


%% training data analysis
figure;
stateCountList = [];
for i = 1:prod(size(stateCount))
    stateCountList = [stateCountList, stateCount(i)];
end
plot(stateCountList,'*')
title('State Counts')

figure;
hold on;
for i = 1:totalEpisodes
    if collisions(i)==1
        plot(i, scores(i),'r*')
    else
        plot(i, scores(i),'b*')
    end
end
title('red: collision; blue: no collision.')

figure;
hold on;
for i = 1:totalEpisodes
    if reachGoal(i)==1
        plot(i, scores(i),'go')
    else
        plot(i, scores(i),'ro')
    end
end
title('green: reach the goal; blue: not reach the goal.')
% plot(scores,'*')

fprintf('Ratio of robots that do not collide during the test: %.3f\n', 1-mean(collisions))
fprintf('Ratio of robots that reach the goal during the test: %.3f\n', mean(reachGoal))
