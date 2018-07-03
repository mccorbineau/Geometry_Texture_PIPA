clear all
close all
clc
rng(3) % noise seed
addpath functions

%% user's choice
sample_name = 'basalt'; % name of the sample: glass, AlCu or basalt
time_max    =  400;    % maximum duration for PIPA after running the initialization algo

%% load data
addpath data
load('H')
load(sample_name);             % load data
x_true     = data.groundtruth; % groundtruth
weight_TV  = data.weight_TV;   % regularization parameter
x_inf      = data.x_inf;       % solution of the decomposition
[m,n]      = size(x_true);     % size of the image
N          = m*n;              % number of pixels

%% apply observation and degradation model for Computed Tomography
chi = 1;                                          % amplitude of uniform noise
y   = H*x_true(:)+chi.*(-1+2.*rand(size(H,1),1)); % tomographic data

%% run PIPA
[x, obj, snr, x_xinf, time] = PIPA(time_max,x_inf,y,chi,weight_TV,H,x_true);

%% plot figures
color=jet(20);
xt = reshape(x(1:N),m,n);
xg = reshape(x(N+1:end),m,n);

%%%% visual results
figure
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.4, 0.8, 0.5]);
%%% texture
subplot(1,4,1)
imagesc(xt); colormap(gray); axis off
title('Texture','Interpreter','latex')
colorbar('Location','southoutside');
set(gca,'fontsize',15)

%%% geometry
subplot(1,4,2)
imagesc(xg); colormap(gray); axis off
title('Geometry','Interpreter','latex')
colorbar('Location','southoutside');
set(gca,'fontsize',15)

%%% reconstruction
subplot(1,4,3)
imagesc(xt+xg); colormap(gray); axis off
title('Reconstruction','Interpreter','latex')
colorbar('Location','southoutside');
set(gca,'fontsize',15)

%%% groundtruth
subplot(1,4,4)
imagesc(x_true); colormap(gray); axis off
title('Groundtruth','Interpreter','latex')
colorbar('Location','southoutside');
set(gca,'fontsize',15)

%%%% quantitative results
figure
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0.1, 0.4, 0.8, 0.5]);
%%% objective function
subplot(1,3,1)
semilogy(time-time(1), obj,'Linewidth',2,'Color',color(2,:))
xlim([0,time(end)-time(1)])
ylim([0.9*min(obj), 1.1*max(obj)])
ylabel({'Criterion'},'Interpreter','latex','fontsize',15)
xlabel({'Time~(s)'},'Interpreter','latex','fontsize',15)
title('Objective solution','Interpreter','latex','fontsize',15)

%%% snr
subplot(1,3,2)
semilogy(time-time(1), snr,'Linewidth',2,'Color',color(2,:))
xlim([0,time(end)-time(1)])
ylim([0.9*min(snr), 1.1*max(snr)])
ylabel({'snr'},'Interpreter','latex','fontsize',15)
xlabel({'Time~(s)'},'Interpreter','latex','fontsize',15)
title('Signal-to-noise ratio','Interpreter','latex','fontsize',15)

%%% distance to solution
subplot(1,3,3)
semilogy(time-time(1), x_xinf,'Linewidth',2,'Color',color(2,:))
xlim([0,time(end)-time(1)])
ylim([0.9*min(x_xinf), 1.1*max(x_xinf)])
ylabel({'$\|x-x_{\infty}\|/\|x_{\infty}\|$'},'Interpreter','latex','fontsize',15)
xlabel({'Time~(s)'},'Interpreter','latex','fontsize',15)
title('Distance to solution','Interpreter','latex','fontsize',15)