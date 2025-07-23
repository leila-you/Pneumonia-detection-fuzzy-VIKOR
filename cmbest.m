% رسم ماتریس درهم‌ریختگی با بهترین دقت
figure;
cm = confusionchart(best_dataTestLabel, best_predictions);

% تغییر رنگ به رنگ‌های گرم
cm.Style = 'filled';  % تنظیم استایل به پرشده
cm.Colormap = hot;  % استفاده از colormap گرم

% تنظیمات تکمیلی برای نمایش بهتر
cm.Title = ['Confusion Matrix - Best Accuracy: ', num2str(best_acc)];
cm.XLabel = 'Predicted Labels';
cm.YLabel = 'True Labels';
cm.FontSize = 12; % تنظیم سایز فونت برای خوانایی بهتر
cm.GridVisible = 'off'; % حذف خطوط شبکه برای نمایش واضح‌تر
