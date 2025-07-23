function [redu,W,List] = fsFisher(X,Y,selectrate)

%Fisher Score, use the N var formulation
%input:   X, the data, each row is an instance
%            Y, the label in 1 2 3 ... format
%           selectrate? 0.1 0.2 0.3...0.9????????????????
%output
%      redu %if selectrate=0.5, redu?top 50% ???
%     W%?????Fisher score??
%     List%??????Fisher score????
%???? [W,List]=fsFisher(c,d)
numC = max(Y);%????
[~, numF] = size(X);%????numF
m=ceil(selectrate*numF);%??????????
W = zeros(1,numF);

% statistic for classes
disp(numC);
disp(class(numC));
% numC = round(numC); % گرد کردن به نزدیک‌ترین عدد صحیح

cIDX = cell(numC,1);%cIDX??????????
n_i = zeros(numC,1);%n_i?????????
for j = 1:numC
    cIDX{j} = find(Y(:)==j);
    n_i(j) = length(cIDX{j});
end

% calculate score for each features
for i = 1:numF%
    temp1 = 0;
    temp2 = 0;
    f_i = X(:,i);
    u_i = mean(f_i);%????????
    
    for j = 1:numC%???numC
        u_cj = mean(f_i(cIDX{j}));
        var_cj = var(f_i(cIDX{j}),1);
        temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
        temp2 = temp2 + n_i(j) * var_cj;
    end
    
%     if temp1 == 0
%         out.W(i) = 0;
%     else
        if temp2 == 0
            W(i) = 1000000;%???0????????????????
        else
            W(i) = temp1/temp2;
        end
%     end
end
[~, List] = sort(W, 'descend');%???????
redu=List(1:m);
% redu=[];
% for i=1:m
%     redu=[redu find(List==i)];
% end
end