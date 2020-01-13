root_dir = pwd;
relative_dir = 'GRUJan-13-10-13/test_predictions.csv';

gru_pred = readtable(fullfile(root_dir, 'GRUJan-13-10-13/test_predictions.csv'));
encoder_pred = readtable(fullfile(root_dir, 'ENCODERJan-13-10-09/test_predictions.csv'));

% Define Colours
color(1,:) = [188,63,69]/255;
color(2,:) = [55,173,241]/255;
color(3,:) = [237,125,49]/255;
color(4,:) = [0,0,0];
color(5,:) = [1,1,0];
color(6,:) = [0,1,1];
color(7,:) = [1,0,1];
color(8,:) = [1,1,1];
color(9,:) = [223,220,202]/255;


figure(1)
clf
subplot(2,2,1);
hold on

plot(encoder_pred.prediction_2014_15, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_15, '-.','color',color(2,:),'linewidth',1.4);
plot(encoder_pred.truth_2014_15,'color',color(3,:),'linewidth',1.4);
title('2014/15')
xlabel('day')
ylabel('ILI rate')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365]);
legend('encoder', 'GRU', 'truth')

subplot(2,2,2);
hold on
plot(encoder_pred.prediction_2014_16, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_16, '-.','color',color(2,:),'linewidth',1.4);
plot(encoder_pred.truth_2014_16,'color',color(3,:),'linewidth',1.4);
title('2015/16')
xlabel('day')
ylabel('ILI rate')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365]);
legend('encoder', 'GRU', 'truth')

subplot(2,2,3);
hold on
plot(encoder_pred.prediction_2014_17, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_17, '-.','color',color(2,:),'linewidth',1.4);
plot(encoder_pred.truth_2014_17,'color',color(3,:),'linewidth',1.4);
title('2016/17')
xlabel('day')
ylabel('ILI rate')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365]);
legend('encoder', 'GRU', 'truth')

subplot(2,2,4);
hold on
plot(encoder_pred.prediction_2014_18, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_18, '-.','color',color(2,:),'linewidth',1.4);
plot(encoder_pred.truth_2014_18,'color',color(3,:),'linewidth',1.4);
title('2018/19')
xlabel('day')
ylabel('ILI rate')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365]);
legend('encoder', 'GRU', 'truth')
