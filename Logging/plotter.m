clear all
root_dir = pwd;

gru_pred = readtable(fullfile(root_dir, 'GRUJan-13-10-13/test_predictions.csv'));
encoder_pred = readtable(fullfile(root_dir, 'ENCODERJan-13-11-03/test_predictions.csv'));
attention_pred = readtable(fullfile(root_dir, 'ATTENTION_day_of_the_year_Jan-13-15-48/test_predictions.csv'));
attention2_pred = readtable(fullfile(root_dir, 'ATTENTIONJan-14-11-11/test_predictions.csv'));


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
plot(attention_pred.prediction_2014_15, ':','color',color(4,:),'linewidth',2);
plot(encoder_pred.truth_2014_15,'color',color(3,:),'linewidth',1.4);
plot(attention2_pred.prediction_2014_15,'color',color(5,:),'linewidth',1.4);
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
ylim([0,60]);
legend('encoder', 'GRU', 'GRU+Attention+DoTY','truth')

subplot(2,2,2);
hold on
plot(encoder_pred.prediction_2014_16, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_16, '-.','color',color(2,:),'linewidth',1.4);
plot(attention_pred.prediction_2014_16, ':','color',color(4,:),'linewidth',2);
plot(encoder_pred.truth_2014_16,'color',color(3,:),'linewidth',1.4);
plot(attention2_pred.prediction_2014_16,'color',color(5,:),'linewidth',1.4);
title('2015/16')
xlabel('day')
ylabel('ILI rate')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
ylim([0,60]);
xlim([0,365]);
legend('encoder', 'GRU', 'GRU+Attention+DoTY','truth')

subplot(2,2,3);
hold on
plot(encoder_pred.prediction_2014_17, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_17, '-.','color',color(2,:),'linewidth',1.4);
plot(attention_pred.prediction_2014_17, ':','color',color(4,:),'linewidth',2);
plot(encoder_pred.truth_2014_17,'color',color(3,:),'linewidth',1.4);
plot(attention2_pred.prediction_2014_17,'color',color(5,:),'linewidth',1.4);
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
ylim([0,60]);
legend('encoder', 'GRU', 'GRU+Attention+DoTY','truth')

subplot(2,2,4);
hold on
plot(encoder_pred.prediction_2014_18, '--','color',color(1,:),'linewidth',1.4);
plot(gru_pred.prediction_2014_18, '-.','color',color(2,:),'linewidth',1.4);
plot(attention_pred.prediction_2014_18, ':','color',color(4,:),'linewidth',2);
plot(encoder_pred.truth_2014_18,'color',color(3,:),'linewidth',1.4);
plot(attention2_pred.prediction_2014_18,'color',color(5,:),'linewidth',1.4);
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
ylim([0,60]);
legend('encoder', 'GRU', 'GRU+Attention+DoTY','truth')


% second figure

encoder = [encoder_pred.prediction_2014_15; encoder_pred.prediction_2014_16; encoder_pred.prediction_2014_17; encoder_pred.prediction_2014_18];
gru = [gru_pred.prediction_2014_15; gru_pred.prediction_2014_16; gru_pred.prediction_2014_17; gru_pred.prediction_2014_18];
attention = [attention_pred.prediction_2014_15; attention_pred.prediction_2014_16; attention_pred.prediction_2014_17; attention_pred.prediction_2014_18];
attention2 = [attention2_pred.prediction_2014_15; attention2_pred.prediction_2014_16; attention2_pred.prediction_2014_17; attention2_pred.prediction_2014_18];
truth = [encoder_pred.truth_2014_15; encoder_pred.truth_2014_16; encoder_pred.truth_2014_17; encoder_pred.truth_2014_18];

figure(2)
clf
subplot(2,1,1)
hold on
plot(encoder, '--','color',color(1,:),'linewidth',1.4);
plot(gru, '-.','color',color(2,:),'linewidth',1.4);
plot(attention, ':','color',color(4,:),'linewidth',2);
% plot(attention2,'color',color(5,:),'linewidth',1.4);
plot(truth,'color',color(3,:),'linewidth',1.4);
title('2014/19')
xlabel('day')
ylabel('ILI rate')
xticks(0:365:365*4)
set(gca,'XMinorTick','on')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365*4]);
ylim([0,60]);
legend('encoder', 'GRU', 'Attention','truth')

subplot(2,1,2)
hold on
plot(encoder-truth, '--','color',color(1,:),'linewidth',1.4);
plot(gru-truth, '-.','color',color(2,:),'linewidth',1.4);
plot(attention-truth, ':','color',color(4,:),'linewidth',2);
% plot(attention2-truth,'color',color(5,:),'linewidth',1.4);

plot(truth,'color',color(3,:),'linewidth',1.4);
title('2014/19 Error Rate')
xlabel('day')
ylabel('ILI error rate')
xticks(0:365:365*4)
set(gca,'XMinorTick','on')
box on
grid on
grid minor
set(gca,'color',color(8,:));
set(gcf,'color',color(8,:));
hold off
xlim([0,365*4]);
legend('error encoder', 'error GRU', 'error GRU+Attention+DoTY','truth')







