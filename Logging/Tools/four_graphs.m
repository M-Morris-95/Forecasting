clear all
root_dir = pwd;
root_dir = root_dir(1:end-6)
cd 'ATTENTIONJan-20-09-35';

figure(4)

clf
hold on

d = dir();
dfolders = d([d(:).isdir]==1);

shape = size(dfolders)-2;
num = shape(1);
base=3;
for i=base:base-1+num
    loc = string(fullfile(dfolders(i).folder,dfolders(i).name, 'test_predictions.csv'));
    pred = readtable(loc);
end

fold = [];
arr_pred = table2array(pred);
Names = pred.Properties.VariableNames;
a=1;
b=1;
c=1;
d=1;
for i = 1:length(Names)
    if contains(string(Names(i)), '14_15') == 1
        fold(1, a,:) = arr_pred(:,i);
        a=a+1;
    elseif contains(string(Names(i)), '14_16') == 1
        fold(2, b,:) = arr_pred(:,i);
        b=b+1;
    elseif contains(string(Names(i)), '14_17') == 1
        fold(3, c,:) = arr_pred(:,i);
        c=c+1;
    elseif contains(string(Names(i)), '14_18') == 1
        fold(4, d,:) = arr_pred(:,i);
        d=d+1;
    end
end

truth = fold(:,2,:);
fold(:,2,:) = [];
mae = [];
rmse = [];
means = [];
for j = 1:4
    subplot(2,2,j)
    hold on
    for i = 1:10
        plot(squeeze(fold(j,i,:)))
    end
    plot(squeeze(truth(j,1,:)), 'linewidth',4, 'color', 'b')
    plot(squeeze(mean(fold(j,:,:))), 'linewidth',4,'color', 'r')
    means(end+1,:) = squeeze(mean(fold(j,:,:)));
    rmse(end+1) = sqrt(mean((means(j,:)-squeeze(truth(j,1,:)).').^2));
    mae(end+1) = mean(abs((means(j,:)-squeeze(truth(j,1,:)).')));
end

[val, idx] = max(means.');
[tru_val, tru_idx] = max(squeeze(truth).');

delay = tru_idx - idx
mae
rmse
cd '/Users/michael/Documents/github/Forecasting/Logging/'

