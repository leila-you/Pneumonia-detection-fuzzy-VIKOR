function [accuracy, Recall, Precision, Fmeasure, predictions] = Classification(method, dataTrain, dataTrainClass, dataTest, dataTestClass)

dataTrainClass = cellstr(num2str(dataTrainClass));
dataTestClass = cellstr(num2str(dataTestClass));

switch method
    case 'svm'
        model = fitcecoc(dataTrain, dataTrainClass);
    case 'knn'
        model = fitcknn(dataTrain, dataTrainClass, 'NumNeighbors', 5);
    case 'dtree'
        model = fitctree(dataTrain, dataTrainClass);
    case 'naiveBayes'
        model = fitcnb(dataTrain, dataTrainClass);
    case 'rf'  % Random Forest
        model = TreeBagger(50, dataTrain, dataTrainClass, 'OOBPrediction', 'on'); % 50 trees
    otherwise
        error('Invalid classification method');
end

predictions = predict(model, dataTest);
% Convert predictions to cell array for consistency with original labels
predictions = cellfun(@str2num, predictions);

accuracy = sum(str2double(dataTestClass) == predictions) / numel(str2double(dataTestClass));
a = str2double(unique(dataTestClass));
actual = str2double(dataTestClass);
pr = predictions;
m = size(a, 1);
de = zeros(m, m);
for i = 1:m
    for j = 1:m
        de(i, j) = sum(pr == a(i) & actual == a(j));
    end
end
Recall = zeros(m, 1);
Precision = zeros(m, 1);
for i = 1:m
    Recall(i, 1) = de(i, i) / sum(de(:, i));
    Precision(i, 1) = de(i, i) / sum(de(i, :));
end
Precision(isnan(Precision)) = 0;
Recall = mean(Recall);
Precision = mean(Precision);
Fmeasure = (2 * Recall * Precision) / (Recall + Precision);

end
