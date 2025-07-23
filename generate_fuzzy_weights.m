function TW = generate_fuzzy_weights(num_criteria, delta)
% Generate fuzzy triangular weights for criteria
% Input:
%   num_criteria - Number of criteria (columns in decision matrix)
%   delta - Small range for fuzzy triangular weights
% Output:
%   TW - Fuzzy triangular weights (num_criteria x 3)

% وزن وسط برای هر معیار (همه معیارها برابر در نظر گرفته می‌شوند)
W_m = ones(num_criteria, 1) * (1 / num_criteria);

% محاسبه مقادیر پایین و بالا
W_l = W_m - delta; % مقدار پایین
W_u = W_m + delta; % مقدار بالا

% ترکیب مقادیر به صورت ماتریس سه‌بعدی
TW = [W_l, W_m, W_u];

end
