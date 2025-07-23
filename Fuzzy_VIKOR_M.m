function Q = Fuzzy_VIKOR_M(D, TW)
% Input:
% D  - Decision Matrix (fuzzy triangular numbers for each criterion)
% TW - Weight matrix (fuzzy triangular weights for each criterion)

[feature_num, num_target, ~] = size(D);

% Extract fuzzy triangular components of the decision matrix
L = D(:, :, 1); % Lower bound
M = D(:, :, 2); % Middle value
U = D(:, :, 3); % Upper bound

% Extract fuzzy triangular components of weights
WL = TW(:, 1); % Lower weight
WM = TW(:, 2); % Middle weight
WU = TW(:, 3); % Upper weight

% Determine the best and worst values for each criterion
f1 = max(U, [], 1); % Best values (upper bound of fuzzy numbers)
f2 = min(L, [], 1); % Worst values (lower bound of fuzzy numbers)

% Validate sizes before loops
if size(WL, 1) ~= num_target || length(f1) ~= num_target || length(f2) ~= num_target
    error('Size mismatch: Check the sizes of input matrices.');
end

% Calculation of fuzzy S (utility measure) and fuzzy R (regret measure)
S = zeros(feature_num, num_target, 3); % Fuzzy S
for i = 1:feature_num
    for j = 1:num_target
        % Compute normalized distance for S
        S(i, j, 1) = WL(j) * (f1(j) - U(i, j)) / (f1(j) - f2(j)); % Lower
        S(i, j, 2) = WM(j) * (f1(j) - M(i, j)) / (f1(j) - f2(j)); % Middle
        S(i, j, 3) = WU(j) * (f1(j) - L(i, j)) / (f1(j) - f2(j)); % Upper
    end
end

% Aggregate fuzzy S and R values
SS = sum(S, 2); % Sum across all criteria
SS = reshape(SS, [feature_num, 3]);

R = max(S, [], 2); % Max across all criteria
R = reshape(R, [feature_num, 3]);

% Calculation of fuzzy VIKOR index (Q)
S1 = max(SS(:, 3)); % Best (upper bound) of aggregated S
S2 = min(SS(:, 1)); % Worst (lower bound) of aggregated S
R1 = max(R(:, 3)); % Best (upper bound) of aggregated R
R2 = min(R(:, 1)); % Worst (lower bound) of aggregated R

v = 0.5; % Balancing parameter
Q = zeros(feature_num, 3); % Initialize fuzzy Q
for i = 1:feature_num
    Q(i, 1) = v * (SS(i, 1) - S2) / (S1 - S2) + (1 - v) * (R(i, 1) - R2) / (R1 - R2); % Lower
    Q(i, 2) = v * (SS(i, 2) - S2) / (S1 - S2) + (1 - v) * (R(i, 2) - R2) / (R1 - R2); % Middle
    Q(i, 3) = v * (SS(i, 3) - S2) / (S1 - S2) + (1 - v) * (R(i, 3) - R2) / (R1 - R2); % Upper
end

end
