clear all
root_dir = pwd;
root_dir = root_dir(1:end-6);
% cd(fullfile(root_dir,'/14 days ahead/GRU_No_DOTY'))
% cd(fullfile(root_dir,'/GRU_14LA_Jan_22_13_54'))
% cd(fullfile(root_dir,'/14 days ahead/GRU_DOTY'))
cd(fullfile(root_dir,'/14 days ahead/ATTN_DOTY'))
% cd(fullfile(root_dir,'/ENCODER_14LA_Jan_24_10_56'))
cd(fullfile(root_dir,'/GRU_14LA_Jan_27_11_11'))
cd(fullfile(root_dir,'/TRANSFORMER_14LA_Jan_28_12_08'))

loc = string(fullfile(pwd, 'test_predictions.csv'));
y_pred = readtable(loc);
y_pred(:,1) = [] ;
Names = y_pred.Properties.VariableNames;
y_pred = table2array(y_pred);

loc = string(fullfile(pwd, 'test_ground_truth.csv'));
y_true = readtable(loc);
y_true(:,1) = [] ;
y_true = table2array(y_true);

figure(2)
clf
hold on
y_15 = zeros(365, 1);
y_16 = zeros(365, 1);
y_17 = zeros(365, 1);
y_18 = zeros(365, 1);

for i = 1:4
    subplot(2,2,i);
    hold on
    plot(y_pred(:,i), 'linewidth',2)
    plot(y_true(:,i), 'linewidth',2)
    
    xlim([0,365])
    legend('prediction', 'truth')
    
    title(string(2013+i) + '/' +string(14+i))
    xlabel('day of the year')
    ylabel('ili rate')
    set(gca,'color',[1,1,1]);
    set(gcf,'color',[1,1,1]);
    box on
    grid on
    grid minor
    
end






cd '/Users/michael/Documents/github/Forecasting/Logging/Tools'


RMSE = zeros(4,1);
MAE = zeros(4,1);
R_ERR = zeros(4,1);



RMSE(1) = sqrt(mean((y_15(:)-y_true(:,1)).^2));
RMSE(2) = sqrt(mean((y_16(:)-y_true(:,2)).^2));
RMSE(3) = sqrt(mean((y_17(:)-y_true(:,3)).^2));
RMSE(4) = sqrt(mean((y_18(:)-y_true(:,4)).^2));

MAE(1) = mean(abs((y_15(:)-y_true(:,1))));
MAE(2) = mean(abs((y_16(:)-y_true(:,2))));
MAE(3) = mean(abs((y_17(:)-y_true(:,3))));
MAE(4) = mean(abs((y_18(:)-y_true(:,4))));

R_ERR(1) = corr(y_15(:), y_true(:,1));
R_ERR(2) = corr(y_16(:), y_true(:,2));
R_ERR(3) = corr(y_17(:), y_true(:,3));
R_ERR(4) = corr(y_18(:), y_true(:,4));

RMSE
MAE
R_ERR

 
