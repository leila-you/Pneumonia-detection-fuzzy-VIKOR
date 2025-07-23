clc;
clear;

addpath('./files');
addpath('./datasets');
datasets = {'output.csv'};

classifiers = {'dtree', 'knn', 'rf', 'svm'};
num_classifiers = length(classifiers);

overall_start_time = tic; % شروع زمان‌گیری کل الگوریتم

for d = 1:length(datasets)
    dataset_name = datasets{d};
    path = strcat(dataset_name);
    dataset = readmatrix('output.csv');
    [~, fNum] = size(dataset);

    fRange = 20:20:2048;
    bucketNum = length(fRange);

    iters = 10;
    time_all = zeros(iters, num_classifiers);

    acc_all = zeros(iters, bucketNum, num_classifiers);
    Recall_all = zeros(iters, bucketNum, num_classifiers);
    Precision_all = zeros(iters, bucketNum, num_classifiers);
    Fmeasure_all = zeros(iters, bucketNum, num_classifiers);
    acc_validation = zeros(iters, bucketNum, num_classifiers);

    best_acc = 0;
    best_predictions = [];
    best_dataTestLabel = [];

    dataset_start_time = tic; % زمان‌گیری پردازش هر دیتاست

    for i = 1:iters
        disp("iters:");
        disp(i);
        [dataTrain, dataTrainLabel, dataTest, dataTestLabel, dataVal, dataValLabel] = SplitDataset2(dataset);
        [~, feature_num] = size(dataTrain);

        % روش‌های انتخاب ویژگی
        M1 = cfs(dataTrain);
        M1(isnan(M1)) = 0; % جایگزینی NaN با صفر
        [~, R1] = sort(M1, 'ascend');

        [redu, W, List] = fsFisher(dataTrain, dataTrainLabel, 0.5);
        M5 = W';
        M5(isnan(M5)) = 0; % جایگزینی NaN با صفر
        [~, R5] = sort(M5, 'descend');

        M13 = llcfs(dataTrain);
        M13(isnan(M13)) = 0; % جایگزینی NaN با صفر
        [~, R7] = sort(M13, 'descend');

        M18 = zeros(feature_num, 1);
        for q = 1:feature_num
            data = dataTrain(:, q);
            answ = mine(data', dataTrainLabel');
            M18(q, 1) = answ.mic;
        end
        M18(isnan(M18)) = 0; % جایگزینی NaN با صفر
        [~, R18] = sort(M18, 'descend');

        % تجمیع رتبه‌بندی‌ها
        P1 = [R5, R7, R1, R18];
        [m, n] = size(P1);
        P8 = [R18, R5, R7, R1];

        P1 = zeros(m, n);
        for q = 1:m
            for v = 1:n
                P1(q, v) = (m + 1) - P8(q, v);
            end
        end

        % روش تجمیع فازی Vikor
        MM = ones(1, 4);
        decision_matrix = P1;
        delta = 0.7;
        fuzzy_decision_matrix = convert_to_fuzzy_matrix(decision_matrix, delta);
        num_criteria = size(decision_matrix, 2);
        delta1 = 0.07;
        TW = generate_fuzzy_weights(num_criteria, delta1);
        E5 = Fuzzy_VIKOR_M(fuzzy_decision_matrix, TW);
        [~, S48] = sort(E5);

        for c = 1:num_classifiers
            classifier = classifiers{c};
            start_time = tic; % زمان‌گیری شروع آموزش

            % آموزش مدل
            train_start_time = tic;
            for j = 1:bucketNum
                % آموزش مدل
                [acc, rec, prec, fmeas, predictions] = Classification(classifier, dataTrain(:, S48(1:fRange(j))), dataTrainLabel, dataTest(:, S48(1:fRange(j))), dataTestLabel);
                % آزمایش مدل
                [acc_val, rec_val, prec_val, fmeas_val, predictions_val] = Classification(classifier, dataTrain(:, S48(1:fRange(j))), dataTrainLabel, dataVal(:, S48(1:fRange(j))), dataValLabel);

                acc_all(i, j, c) = acc;
                Recall_all(i, j, c) = rec;
                Precision_all(i, j, c) = prec;
                Fmeasure_all(i, j, c) = fmeas;
                acc_validation(i, j, c) = acc_val;

                if acc > best_acc
                    best_acc = acc;
                    best_predictions = predictions;
                    best_dataTestLabel = dataTestLabel;
                end
            end
            train_time = toc(train_start_time); % زمان‌گیری آموزش
            time_all(i, c) = train_time; % ذخیره زمان آموزش
        end

        % زمان‌گیری آزمایش
        test_start_time = tic;
        % کد آزمایش
        % (فرض کنید آزمایش مدل در اینجا انجام می‌شود)
        test_time = toc(test_start_time); % زمان‌گیری آزمایش
        disp(['Testing Time for Iteration ', num2str(i), ': ', num2str(test_time)]);

    end

    dataset_time_elapsed = toc(dataset_start_time); % زمان کل پردازش این دیتاست
    disp("Total Time for Dataset: "); disp(dataset_time_elapsed);

    % محاسبه میانگین‌ها با حذف NaN
    for c = 1:num_classifiers
        disp(["Results for ", classifiers{c}]);
        mean_acc = nanmean(acc_all(:,:,c), 1);
        mean_recall = nanmean(Recall_all(:,:,c), 1);
        mean_precision = nanmean(Precision_all(:,:,c), 1);
        mean_fmeasure = nanmean(Fmeasure_all(:,:,c), 1);
        mean_acc_validation = nanmean(acc_validation(:,:,c), 1);
        mean_time = nanmean(time_all(:, c));

        disp("Mean Accuracy:"); disp(mean_acc);
        disp("Mean Recall:"); disp(mean_recall);
        disp("Mean Precision:"); disp(mean_precision);
        disp("Mean Fmeasure:"); disp(mean_fmeasure);
        disp("Mean Validation Accuracy:"); disp(mean_acc_validation);
        disp("Mean Execution Time:"); disp(mean_time);
    end
end

overall_time_elapsed = toc(overall_start_time); % زمان کل اجرای کل الگوریتم
disp("Total Execution Time for All Datasets: "); disp(overall_time_elapsed);

% ذخیره نتایج به یک فایل
save('execution_results.mat', 'acc_all', 'Recall_all', 'Precision_all', 'Fmeasure_all');
