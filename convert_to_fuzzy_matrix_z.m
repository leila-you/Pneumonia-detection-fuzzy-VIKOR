function fuzzy_matrix = convert_to_fuzzy_matrix_z(decision_matrix, delta)
% Convert a normal decision matrix to a fuzzy decision matrix
% Input:
%   decision_matrix - Normal decision matrix (m x n)
%   delta - Small range for fuzzy triangular numbers
% Output:
%   fuzzy_matrix - Fuzzy decision matrix (m x n x 3)
[m, n] = size(decision_matrix); % ابعاد ماتریس تصمیم
fuzzy_matrix = zeros(m, n, 4); % ماتریس فازی ذوزنقه‌ای
delta_lower = 0.1; % ضریب برای حد پایین
delta_upper = 0.2; % ضریب برای حد بالا

for i = 1:m
    for j = 1:n
        value = decision_matrix(i, j);
        fuzzy_matrix(i, j, 1) = value - delta_upper; % a
        fuzzy_matrix(i, j, 2) = value - delta_lower; % b
        fuzzy_matrix(i, j, 3) = value + delta_lower; % c
        fuzzy_matrix(i, j, 4) = value + delta_upper; % d
    end
end
