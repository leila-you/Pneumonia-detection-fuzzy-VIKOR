dataTestLabel = categorical(dataTestLabel);   % تبدیل برچسب‌های واقعی به عددی
pred = categorical(predictions); 
cm = confusionmat(dataTestLabel, pred);
% % نمایش ماتریس آشفتگی
% disp('Confusion Matrix:');
% %disp(confMat);
% % رسم ماتریس سردرگمی
% figure;
% imagesc(confMat);
% colorbar;
% title('Confusion Matrix');
% xlabel('Predicted Class');
% ylabel('True Class');


figure;
imagesc(cm);  % رسم ماتریس سردرگمی به صورت تصویر

% گام 3: افزودن تعداد به داخل خانه‌ها
for i = 1:size(cm, 1)
    for j = 1:size(cm, 2)
        % نوشتن مقدار تعداد داخل هر خانه
        text(j, i, num2str(cm(i,j)), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle', 'Color', 'white', 'FontSize', 12);
    end
end

% گام 4: افزودن برچسب‌ها و عنوان‌ها
colorbar;  % نمایش نوار رنگ
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
colormap jet;  % انتخاب رنگ‌بندی

