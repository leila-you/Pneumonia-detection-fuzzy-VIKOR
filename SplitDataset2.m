% function [dataTrain, dataTrainClass, dataTest, dataTestClass, dataVal, dataValClass] = SplitDataset(dataset)
% 
% % جداسازی ویژگی‌ها و کلاس‌ها
% instances = dataset(:, 1:end-2);   % ویژگی‌ها
% classes = dataset(:, end-1);        % کلاس‌ها
% 
% % تعداد کل داده‌ها
% totalData = size(instances, 1);
% 
% % محاسبه تعداد نمونه‌ها برای هر کلاس در کل داده‌ها
% disp('Total Data Class Distribution:');
% uniqueClasses = unique(classes);  % یافتن کلاس‌های منحصر به فرد
% for i = 1:length(uniqueClasses)
%     classCount = sum(classes == uniqueClasses(i));  % شمارش تعداد نمونه‌ها برای هر کلاس
%     fprintf('Class %d: %d instances\n', uniqueClasses(i), classCount);  % نمایش تعداد نمونه‌ها
% end
% 
% % Cross-validation (train: 75%, test: 15%, validation: 10%)
% cv = cvpartition(totalData, 'HoldOut', 0.15);  % 15% برای تست
% idxTest = cv.test;
% 
% % داده‌های تست و باقی‌مانده (85% می‌شود)
% dataTest = instances(idxTest, :);
% dataTestClass = classes(idxTest, :);
% dataRemaining = instances(~idxTest, :);
% classesRemaining = classes(~idxTest, :);
% 
% % از داده‌های باقی‌مانده 10% را برای ولیدیشن می‌گیریم
% cv2 = cvpartition(size(dataRemaining, 1), 'HoldOut', 0.1176); % 10% از 85% داده‌ها که معادل 10% از کل داده‌ها می‌شود
% idxVal = cv2.test;
% 
% % داده‌های ولیدیشن و آموزش
% dataVal = dataRemaining(idxVal, :);
% dataValClass = classesRemaining(idxVal, :);
% dataTrain = dataRemaining(~idxVal, :);
% dataTrainClass = classesRemaining(~idxVal, :);
% 
% % نمایش تعداد داده‌ها در هر بخش
% numTrain = size(dataTrain, 1);
% numTest = size(dataTest, 1);
% numVal = size(dataVal, 1);
% 
% % چاپ تعداد کل داده‌ها و داده‌ها در هر بخش
% fprintf('Total data: %d\n', totalData);
% fprintf('Training data: %d\n', numTrain);
% fprintf('Testing data: %d\n', numTest);
% fprintf('Validation data: %d\n', numVal);
% 
% % محاسبه تعداد نمونه‌ها برای هر کلاس در هر بخش
% 
% % برای داده‌های آموزشی
% disp('Training Data Class Distribution:');
% uniqueTrainClasses = unique(dataTrainClass);
% for i = 1:length(uniqueTrainClasses)
%     classCount = sum(dataTrainClass == uniqueTrainClasses(i));
%     fprintf('Class %d: %d instances\n', uniqueTrainClasses(i), classCount);
% end
% 
% % برای داده‌های تست
% disp('Testing Data Class Distribution:');
% uniqueTestClasses = unique(dataTestClass);
% for i = 1:length(uniqueTestClasses)
%     classCount = sum(dataTestClass == uniqueTestClasses(i));
%     fprintf('Class %d: %d instances\n', uniqueTestClasses(i), classCount);
% end
% 
% % برای داده‌های ولیدیشن
% disp('Validation Data Class Distribution:');
% uniqueValClasses = unique(dataValClass);
% for i = 1:length(uniqueValClasses)
%     classCount = sum(dataValClass == uniqueValClasses(i));
%     fprintf('Class %d: %d instances\n', uniqueValClasses(i), classCount);
% end
% 
% end




function [dataTrain, dataTrainClass, dataTest, dataTestClass, dataVal, dataValClass, trainIdx, testIdx, valIdx] = SplitDataset2(dataset)

% جداسازی ویژگی‌ها و کلاس‌ها
instances = dataset(:, 1:end-2);   % ویژگی‌ها
classes = dataset(:, end-1);        % کلاس‌ها

% تعداد کل داده‌ها
totalData = size(instances, 1);

% Cross-validation (train: 75%, test: 15%, validation: 10%)
cv = cvpartition(totalData, 'HoldOut', 0.15);  % 15% برای تست
testIdx = cv.test;  % ایندکس نمونه‌های تست

dataTest = instances(testIdx, :);
dataTestClass = classes(testIdx, :);

dataRemaining = instances(~testIdx, :);
classesRemaining = classes(~testIdx, :);

cv2 = cvpartition(size(dataRemaining, 1), 'HoldOut', 0.1176); % حدود 10% برای validation از داده‌های باقی‌مانده
valIdxInRemaining = cv2.test;

dataVal = dataRemaining(valIdxInRemaining, :);
dataValClass = classesRemaining(valIdxInRemaining, :);

dataTrain = dataRemaining(~valIdxInRemaining, :);
dataTrainClass = classesRemaining(~valIdxInRemaining, :);

% محاسبه ایندکس نمونه‌ها نسبت به دیتاست اصلی

% ایندکس نمونه‌های باقی‌مانده نسبت به دیتاست اصلی
remainingIdx = find(~testIdx);

% ایندکس ولیدیشن نسبت به دیتاست اصلی
valIdx = remainingIdx(valIdxInRemaining);

% ایندکس آموزش نسبت به دیتاست اصلی
trainIdx = remainingIdx(~valIdxInRemaining);

% نمایش آمار (اختیاری)
fprintf('Total data: %d\n', totalData);
fprintf('Training data: %d\n', size(dataTrain, 1));
fprintf('Testing data: %d\n', size(dataTest, 1));
fprintf('Validation data: %d\n', size(dataVal, 1));

end
