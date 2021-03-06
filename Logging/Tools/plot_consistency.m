clear all
root_dir = '/Users/michael/Documents/Forecasting Old Logs/consistency';

% cd(fullfile(root_dir,'/ATTENTION_14LA_Jan_21_14_30')) 
% cd(fullfile(root_dir,'/ATTENTION_Feb_17_uniform')) 
cd(fullfile(root_dir,'/ATTENTION_Feb_17_normal')) 

% cd(fullfile(root_dir,'/ENCODER_14LA_Jan_24_11_14')) 
% cd(fullfile(root_dir,'/GRU_14LA_Jan_22_13_54')) 
% cd(fullfile(root_dir,'/MODENC_14LA_Jan_29_10_04')) 

% Define Colours
color(1,:) = [188,63,69]/255;
color(2,:) = [55,173,241]/255;
color(3,:) = [237,125,49]/255;

loc = string(fullfile(pwd, 'test_predictions.csv'));
y_pred = readtable(loc);
y_pred(:,1) = [] ;
Names = y_pred.Properties.VariableNames;
y_pred = table2array(y_pred);

loc = string(fullfile(pwd, 'test_ground_truth.csv'));
y_true = readtable(loc);
y_true(:,1) = [] ;
y_true = table2array(y_true);


K = floor(length(Names)/4);


y_15 = zeros(366, K);
y_16 = zeros(366, K);
y_17 = zeros(366, K);
y_18 = zeros(366, K);
a = 1;
b = 1;
c = 1;
d = 1;
plot_mean = 1;
figure(1)
clf

for i = 1:K*4
    if contains(string(Names(i)), '14_15') == 1
        subplot(2,2,1);
        hold on
        plot(y_pred(:,i),'HandleVisibility','off')
        y_15(:,a) = y_pred(:,i);
        a = a+1;
        title('2014/15')
        if a == K+1
            if plot_mean == 1
                plot(mean(y_15.'), '--','linewidth', 3, 'DisplayName','Mean', 'color',color(1,:))      
            end
        end
    elseif contains(string(Names(i)), '15_16') == 1
        subplot(2,2,2);
        hold on
        plot(y_pred(:,i),'HandleVisibility','off')
        y_16(:,b) = y_pred(:,i);
        b = b+1;
        title('2015/16')
        if b == K+1
            if plot_mean == 1
            plot(mean(y_16.'), '--','linewidth', 3, 'DisplayName','Mean', 'color',color(1,:))  
            end
        end
    elseif contains(string(Names(i)), '16_17') == 1
        subplot(2,2,3);
        hold on
        plot(y_pred(:,i),'HandleVisibility','off')
        y_17(:,c) = y_pred(:,i);
        c = c+1;
        title('2016/17')
        if c == K+1
            if plot_mean == 1
            plot(mean(y_17.'), '--','linewidth', 3, 'DisplayName','Mean', 'color',color(1,:))  
            end
        end
    elseif contains(string(Names(i)), '17_18') == 1
        subplot(2,2,4);
        hold on
        plot(y_pred(:,i),'HandleVisibility','off')
        y_18(:,d) = y_pred(:,i);
        d = d+1;
        title('2017/18')
        if d == K+1
            if plot_mean == 1
            plot(mean(y_18.'), '--','linewidth', 3, 'DisplayName','Mean', 'color',color(1,:))  
            end
        end
    end
end
for i = 1:4
    subplot(2,2,i)
    hold on
    plot(y_true(:,i), '-.','linewidth', 3, 'DisplayName','Ground Truth', 'color',color(2,:))  
    xlim([0,365])
%     legend('truth', 'mean')
    legend()
    hold off
    xlabel('day of the year')
    ylabel('ili rate')
    set(gca,'color',[1,1,1]);
    set(gcf,'color',[1,1,1]);
    box on
    grid on
    grid minor

end
suptitle('attention model consistency with normally distibuted weights')





cd '/Users/michael/Documents/github/Forecasting/Logging/Tools'

R = zeros(4,45);
RMSE = zeros(4,K);
MAE = zeros(4,K);
R_ERR = zeros(4,K);
count = 0;

for j = 1:K
    for k = j+1:K
        count = count+1;
        R(1,count) = corr(y_15(:,j), y_15(:,k));
        R(2,count) = corr(y_16(:,j), y_16(:,k));
        R(3,count) = corr(y_17(:,j), y_17(:,k));
        R(4,count) = corr(y_18(:,j), y_18(:,k));
    end
        RMSE(1,j) = sqrt(nanmean((y_15(:,j)-y_true(:,1)).^2));
        RMSE(2,j) = sqrt(nanmean((y_16(:,j)-y_true(:,2)).^2));
        RMSE(3,j) = sqrt(nanmean((y_17(:,j)-y_true(:,3)).^2));
        RMSE(4,j) = sqrt(nanmean((y_18(:,j)-y_true(:,4)).^2));
        
        MAE(1,j) = nanmean(abs(y_15(:,j)-y_true(:,1)));
        MAE(2,j) = nanmean(abs(y_16(:,j)-y_true(:,2)));
        MAE(3,j) = nanmean(abs(y_17(:,j)-y_true(:,3)));
        MAE(4,j) = nanmean(abs(y_18(:,j)-y_true(:,4)));
        
        R_ERR(1,j) = corr(y_15(:,j), y_true(:,1));
        R_ERR(2,j) = corr(y_16(:,j), y_true(:,2));
        R_ERR(3,j) = corr(y_17(:,j), y_true(:,3));
        R_ERR(4,j) = corr(y_18(:,j), y_true(:,4));
end

 max(R.') - min(R.');
 max(RMSE.')- min(RMSE.');
 max(MAE.')- min(MAE.');
max(R_ERR.')- min(R_ERR.');



for j = 1:k
    if y_15(5,j)==0
        idx = j;
        continue
    end
    for k = j+1:k
        count = count+1;
        R(1,count) = corr(y_15(:,j), y_15(:,k));
        R(2,count) = corr(y_16(:,j), y_16(:,k));
        R(3,count) = corr(y_17(:,j), y_17(:,k));
        R(4,count) = corr(y_18(:,j), y_18(:,k));
    end
        RMSE(1,j) = sqrt(nanmean((y_15(:,j)-y_true(:,1)).^2));
        RMSE(2,j) = sqrt(nanmean((y_16(:,j)-y_true(:,2)).^2));
        RMSE(3,j) = sqrt(nanmean((y_17(:,j)-y_true(:,3)).^2));
        RMSE(4,j) = sqrt(nanmean((y_18(:,j)-y_true(:,4)).^2));
        
        MAE(1,j) = nanmean(abs((y_15(:,j)-y_true(:,1))));
        MAE(2,j) = nanmean(abs((y_16(:,j)-y_true(:,2))));
        MAE(3,j) = nanmean(abs((y_17(:,j)-y_true(:,3))));
        MAE(4,j) = nanmean(abs((y_18(:,j)-y_true(:,4))));
        
        R_ERR(1,j) = corr(y_15(:,j), y_true(:,1));
        R_ERR(2,j) = corr(y_16(:,j), y_true(:,2));
        R_ERR(3,j) = corr(y_17(:,j), y_true(:,3));
        R_ERR(4,j) = corr(y_18(:,j), y_true(:,4));
end
% RMSE(:,idx) = [];
% R_ERR(:,idx) = [];
% MAE(:,idx) = [];


 max(R.') - min(R.');
 max(MAE.')- min(MAE.');
 max(RMSE.')- min(RMSE.');
 max(R_ERR.')- min(R_ERR.');

 mean(MAE.');
 mean(RMSE.');
 mean(R_ERR.');

 min(MAE.');
 min(RMSE.');
 max(R_ERR.');
 
 sum = 0;
 for i = 1:23
     sum = sum + (23-i)/365;
 end