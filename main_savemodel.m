clc;
clear;

addpath('./files');
addpath('./datasets');
mkdir('saved_models');  % ایجاد پوشه ذخیره مدل‌ها

datasets = {'output.csv'};
classifiers = {'dtree', 'knn', 'rf', 'svm'};
num_classifiers = length(classifiers);

overall_start_time = tic;

for d = 1:length(datasets)
    dataset_name = datasets{d};
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

    dataset_start_time = tic;

    for i = 1:iters
        disp("iters:");
        disp(i);
        [dataTrain, dataTrainLabel, dataTest, dataTestLabel, dataVal, dataValLabel] = SplitDataset2(dataset);
        [~, feature_num] = size(dataTrain);

        M1 = cfs(dataTrain); M1(isnan(M1)) = 0; [~, R1] = sort(M1, 'ascend');
        [redu, W, List] = fsFisher(dataTrain, dataTrainLabel, 0.5); M5 = W'; M5(isnan(M5)) = 0; [~, R5] = sort(M5, 'descend');
        M13 = llcfs(dataTrain); M13(isnan(M13)) = 0; [~, R7] = sort(M13, 'descend');

        M18 = zeros(feature_num, 1);
        for q = 1:feature_num
            data = dataTrain(:, q);
            answ = mine(data', dataTrainLabel');
            M18(q, 1) = answ.mic;
        end
        M18(isnan(M18)) = 0; [~, R18] = sort(M18, 'descend');

        P8 = [R18, R5, R7, R1];
        [m, n] = size(P8);
        P1 = zeros(m, n);
        for q = 1:m
            for v = 1:n
                P1(q, v) = (m + 1) - P8(q, v);
            end
        end

        delta = 0.7;
        fuzzy_decision_matrix = convert_to_fuzzy_matrix(P1, delta);
        delta1 = 0.07;
        TW = generate_fuzzy_weights(n, delta1);
        E5 = Fuzzy_VIKOR_M(fuzzy_decision_matrix, TW);
        [~, S48] = sort(E5);

        for c = 1:num_classifiers
            classifier = classifiers{c};
            train_start_time = tic;

            for j = 1:bucketNum
                [acc, rec, prec, fmeas, predictions, trained_model] = Classification(classifier, ...
                    dataTrain(:, S48(1:fRange(j))), dataTrainLabel, ...
                    dataTest(:, S48(1:fRange(j))), dataTestLabel);

                [acc_val, rec_val, prec_val, fmeas_val, ~, ~] = Classification(classifier, ...
                    dataTrain(:, S48(1:fRange(j))), dataTrainLabel, ...
                    dataVal(:, S48(1:fRange(j))), dataValLabel);

                acc_all(i, j, c) = acc;
                Recall_all(i, j, c) = rec;
                Precision_all(i, j, c) = prec;
                Fmeasure_all(i, j, c) = fmeas;
                acc_validation(i, j, c) = acc_val;

                % ذخیره مدل فعلی
                model_filename = sprintf('model_%s_iter%d_bucket%d.mat', classifier, i, j);
                save(fullfile('saved_models', model_filename), 'trained_model');

                % ثبت بهترین مدل
                if acc > best_acc
                    best_acc = acc;
                    best_predictions = predictions;
                    best_dataTestLabel = dataTestLabel;
                    best_model = trained_model;
                    best_model_name = sprintf('best_model_%s_iter%d.mat', classifier, i);
                end
            end

            train_time = toc(train_start_time);
            time_all(i, c) = train_time;
        end

        % ذخیره بهترین مدل در این تکرار
        if exist('best_model', 'var')
            save(fullfile('saved_models', best_model_name), 'best_model');
        end
    end

    dataset_time_elapsed = toc(dataset_start_time);
    disp("Total Time for Dataset: "); disp(dataset_time_elapsed);

    for c = 1:num_classifiers
        disp(["Results for ", classifiers{c}]);
        disp("Mean Accuracy:"); disp(nanmean(acc_all(:,:,c), 1));
        disp("Mean Recall:"); disp(nanmean(Recall_all(:,:,c), 1));
        disp("Mean Precision:"); disp(nanmean(Precision_all(:,:,c), 1));
        disp("Mean Fmeasure:"); disp(nanmean(Fmeasure_all(:,:,c), 1));
        disp("Mean Validation Accuracy:"); disp(nanmean(acc_validation(:,:,c), 1));
        disp("Mean Execution Time:"); disp(nanmean(time_all(:, c)));
    end
end

overall_time_elapsed = toc(overall_start_time);
disp("Total Execution Time for All Datasets: "); disp(overall_time_elapsed);

save('execution_results.mat', 'acc_all', 'Recall_all', 'Precision_all', 'Fmeasure_all');
