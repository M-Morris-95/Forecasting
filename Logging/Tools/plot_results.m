clear all
root_dir = pwd;
root_dir = root_dir(1:end-6);

root_dir = '/Users/michael/Documents/Forecasting Old Logs/consistency';
% cd(fullfile(root_dir,'/14 days ahead/GRU_No_DOTY'))
% cd(fullfile(root_dir,'/GRU_14LA_Jan_22_13_54'))
% cd(fullfile(root_dir,'/14 days ahead/GRU_DOTY'))
cd(fullfile(root_dir,'/14 days ahead/ATTN_DOTY'))
% cd(fullfile(root_dir,'/ENCODER_14LA_Jan_24_10_56'))
cd(fullfile(root_dir,'/consistency/MODENC_14LA_Jan_29_10_04'))
cd(fullfile(root_dir,'/SIMPLE_14LA_Jan_31_14_19'))
cd(fullfile(root_dir,'/GRU_21LA_Feb_03_16_30'))

% cd(fullfile(root_dir,'/Simple_None'))
% cd(fullfile(root_dir,'/Simple/Simple_doty'))
% cd(fullfile(root_dir,'/Simple_weather'))
% cd(fullfile(root_dir,'/Simple_all'))

loc = string(fullfile(pwd, 'test_predictions.csv'));
y_pred = readtable(loc);
y_pred(:,1) = [] ;
Names = y_pred.Properties.VariableNames;
y_pred = table2array(y_pred);

loc = string(fullfile(pwd, 'test_ground_truth.csv'));
y_true = readtable(loc);
y_true(:,1) = [] ;
y_true = table2array(y_true);

K = length(y_true(1,:))/4;



figure(1)
clf
hold on

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

figure(2)
clf
for i = 1:K*4
    if contains(string(Names(i)), '14_15') == 1
        subplot(2,2,1);
        hold on
        plot(y_pred(:,i),'HandleVisibility','on')
    elseif contains(string(Names(i)), '15_16') == 1
        subplot(2,2,2);
        hold on
        plot(y_pred(:,i),'HandleVisibility','on')
    elseif contains(string(Names(i)), '16_17') == 1
        subplot(2,2,3);
        hold on
        plot(y_pred(:,i),'HandleVisibility','on')
    elseif contains(string(Names(i)), '17_18') == 1
        subplot(2,2,4);
        hold on
        plot(y_pred(:,i),'HandleVisibility','on')
    end
end
for i = 1:4
    subplot(2,2,i);
    hold on
    plot(y_true(:,i), 'linewidth',2)
    
    xlim([0,365])


    legend('L1= 0.1 L2= 0.0 ','L1= 0.0 L2= 0.0 ','L1= 0.0 L2= 0.001 ','L1= 0.001 L2= 0.001 ','L1= 0.01 L2= 0.001 ','L1= 0.1 L2= 0.001 ','L1= 0.001 L2= 0.01 ','L1= 0.0 L2= 0.0 ','L1= 0.01 L2= 0.01 ','L1= 0.001 L2= 0.1 ','L1= 0.1 L2= 0.01 ','L1= 0.0 L2= 0.0 ','L1= 0.01 L2= 0.1 ','L1= 0.0 L2= 0.0 ','L1= 0.0 L2= 0.0 ','L1= 0.1 L2= 0.1 ','truth')
    
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


Error = y_pred-y_true;



T_RMSE = sqrt(mean(Error.^2));
T_MAE = mean(abs(Error));
T_R = zeros(length(y_pred(1,:)),1);
for i = 1:length(y_pred(1,:))
    T_R(i) = corr(y_pred(:,i), y_true(:,i));
end

RMSE = zeros(10,4);
MAE = zeros(10,4);
R = zeros(10,4);
count = 1;
for i = 1:length(y_pred(1,:))
    for j = 1:4
        if mod(i+1,4) == 0
            RMSE(count,j) = T_RMSE(i);
            MAE(count,j) = T_MAE(i);
            R(count,j) = T_R(i);
            if j == 4
                count = count+1;
            end
        end
    end
end





 
