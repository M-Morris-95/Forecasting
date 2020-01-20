clear all
root_dir = pwd;
cd 'ATTENTIONJan-17-15-25'
pwd

d = dir();
dfolders = d([d(:).isdir]==1);
figure(1)
clf
hold on
for i=3:12
    loc = string(fullfile(dfolders(i).folder,dfolders(i).name, 'test_predictions.csv'));
    pred = readtable(loc);
end

pred = table2array(pred);
truth = pred(:,3);
pred(:,3) = [];
pred(:,1) = [];
thicc= 5;
med_thicc = 3;
plot(truth, 'linewidth',thicc)
plot(mean(pred, 2), 'linewidth',thicc)
% plot(mean(pred, 2)-0.5*std(pred, 1, 2), 'linewidth',med_thicc)
% plot(mean(pred, 2)+0.5*std(pred, 1, 2), 'linewidth',med_thicc)


legend('Truth', 'Average');


for i =1:10
    plot(pred(:,i))
end
cd '/Users/michael/Documents/github/Forecasting/Logging/'


% gru_pred = readtable(fullfile(root_dir, '14 days ahead/GRUJan-13-10-13/test_predictions.csv'));