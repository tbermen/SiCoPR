function port_order = auto_port_order(sch, F, flip_victim)

%% Automatically determine port order
if nargin < 3
    flip_victim = 0;
end

MinThruEnergy = 0.1;

num_ports = size(sch,3);
if num_ports ~= 4
    error('Auto Port Order routine only works for 4 port S-parameters');
end

% Observe 1st frequency point
LowFreq_Matrix = abs(squeeze(sch(1,:,:)));
Raw_LowFreq_Matrix = LowFreq_Matrix;

% Ignore RL:  set diagonal terms to 0
LowFreq_Matrix = LowFreq_Matrix - diag(diag(LowFreq_Matrix));

% Force Reciprocity by taking the max element of any reciprical term
upper_triangle =  triu(LowFreq_Matrix);
lower_triangle = transpose(tril(LowFreq_Matrix));
max_matrix = max(upper_triangle,lower_triangle);
LowFreq_Matrix = max_matrix + transpose(triu(max_matrix));

% Find connected ports by observing max value in each column
% (Could also use each row since reciprocity has been enforced)
ConnectedPorts = zeros(1,4);
for k = 1:4
    [max_value, ConnectedPorts(k)] = max(LowFreq_Matrix(:,k));
    if max_value < MinThruEnergy
        error('Unable to determine port connections:  Low Energy');
    end
end

% Force that connected ports agree with each other
% That is:  if port 1 connects to port 2, port 2 should connect with port 1
for k = 1:4
    my_connection = ConnectedPorts(k);
    other_connection = ConnectedPorts(my_connection);
    if other_connection ~= k
        error('Unable to determine port connections:  Ambiguous connections');
    end
end

% Set initial port order
port_order = zeros(1,4);
port_order(1) = 1;
% 3rd index is the connection to port 1
port_order(3) = ConnectedPorts(1);
% 2nd index is the minimum port not part of port 1 connection
port_order(2) = min(setdiff(1:4, [1 ConnectedPorts(1)]));
% 4th index is the maximum port not part of port 1 connection
port_order(4) = max(setdiff(1:4, [1 ConnectedPorts(1)]));

if ConnectedPorts(port_order(2)) ~= port_order(4)
    error('Unable to determine port connections:  Ambiguous connections');
end

% Determine if port_order([2 4]) should be swapped by observing phase delay vs. Port 1
% In general, a large delta in phase delay reveals whether a port is on the near side or far side of a reference port
try
    TxN = port_order(2);
    RxN = port_order(4);

    % Check phase delay to see Rx pair should be swapped
    % Use the connection to port 1 with the largest magnitude
    if Raw_LowFreq_Matrix(TxN,1) > Raw_LowFreq_Matrix(1, TxN)
        vector1 = squeeze(sch(:,TxN,1));
    else
        vector1 = squeeze(sch(:,1,TxN));
    end
    if Raw_LowFreq_Matrix(RxN,1) > Raw_LowFreq_Matrix(1, RxN)
        vector2 = squeeze(sch(:,RxN,1));
    else
        vector2 = squeeze(sch(:,1,RxN));
    end

    % Check if DC needs to be removed for phase delay calc
    F = F(:);
    if F(1) == 0
        vector1 = vector1(2:end);
        vector2 = vector2(2:end);
        F = F(2:end);
    end

    % phase delay of port connected to port 1
    phase_delay1 = -1*unwrap(angle(vector1)) ./ (F*2*pi);
    phase_delay2 = -1*unwrap(angle(vector2)) ./ (F*2*pi);

    % make phase delay scalar by using the mean on the 0.25:0.75 range of frequency vector
    quarter_size = round(length(F)/4);
    three_quarter_size = round(length(F)*3/4);
    mean_phase_delay1 = mean(phase_delay1(quarter_size:three_quarter_size));
    mean_phase_delay2 = mean(phase_delay2(quarter_size:three_quarter_size));

    % Set a confidence check.  Fext phase delay should be 2x Next phase delay to make a decision
    % 2x is arbitrary.  Any better conditions?
    if max(mean_phase_delay1, mean_phase_delay2) > min(mean_phase_delay1, mean_phase_delay2)*2
        % The maximum phase delay is the far side.  It should always be the 2nd phase delay
        % If far_side = 1, swap the ports
        [~, far_side] = max([mean_phase_delay1, mean_phase_delay2]);
        if far_side == 1
            port_order([2 4]) = port_order([4 2]);
        end
    else
        fprintf('Did not use phase delay in auto-port discovery since the phase delay of Near End and Far End are similar\n');
    end
catch ME_msg
    fprintf('%s\n',ME_msg.message);
    for j=1:length(ME_msg.stack)
        this_stack=ME_msg.stack(j);
        fprintf(2,'Error in %s @ Line %d\n',this_stack.name,this_stack.line);
    end
    fprintf('Unable to use phase delay to determine port order\n');
end

% Flip which side is the Rx side
% In most cases, this doesn't matter, but there is only 1 valid Rx side for an Active Device
if flip_victim
    port_order = port_order([3 4 1 2]);
end

fprintf('Auto Port Order: [%s]\n', num2str(port_order));
