function [x, obj_vec, snr_vec, x_x_inf_vec, time_vec] = PIPA(time_max,x_inf,y,chi,weightTV,H,x_true)

%====================================================================
% Associated citation:
%
% M.-C. Corbineau, E. Chouzenoux, and J.-C. Pesquet. Geometry-texture
% decomposition/reconstruction using a proximal interior point algorithm.
% In Proceedings of the 10th IEEE Sensor Array and Multichannel Signal
% Processing Workshop (SAM 2018), Sheffield, UK, 8-11 July 2018.
%
% Kindly report any suggestions or corrections to
% marie-caroline.corbineau@centralesupelec.fr
%
% Input:  time_max: maximal duration after initialization
%         x_inf: solution (used only to compute the distance to solution)    
%         y: observed data
%         chi: measurement uncertainty
%         weightTV: regularization parameter weighting the total variation term 
%         H: observation operator (Radon transform)
%         x_true: ground-truth (used only to compute the signal-to-noise ratio)
%
% Output: x: decomposition, x(1:N)=texture, x(N+1:end)=geometry 
%         obj_vec: objective function value for every iteration
%         snr_vec: signal-to-noise ratio for every iteration
%         x_x_inf_vec: distance to solution for every iteration
%         time_vec: time for every iteration (includes duration for initialization)
%
% Solves P_I:                 minimize      0.5*||F(xt)||^2_2 + weightTV*TV(xg)  
%                     for (xt and xg in R^N)
%                             subject to   -chi <= H*(xt+xg)-y <= chi
%                                           0 <= xt+xg <= 1
%
% where F is an edge detection operator derived form the Laplacian.
%====================================================================

%%%%%%%%%%%% algorithm parameters 
maxiter_backtrack  = 100;                     % maximum number of iterations for the inner loop
mu                 = 1;                       % initialization of barrier parameter
precision          = max(min(1e-5,mu),1e-7);  % precision for computing the proximity operator of TV
gamma_max          = 1.5;                     % maximal stepsize
theta_backtrack    = 0.8;                     % granularity of backtracking search
delta_backtrack    = 0.99;                    % Wolfe parameter of backtracking search     
eps1               = 1;                       % accuracy for stopping criterion 1
eps1_rule          = [100 200  500 ;...
                      0.1 0.07 0.03];  % rule for changing eps1 (line 1: iteration, line 2: value)
eps2               = 1e10;             % accuracy for stopping criterion 2
eps3               = 1e7;              % accuracy for stopping criterion 3
rho                = 1.01;             % constant for geometric decrease of barrier parameter 
rho_rule           = [10000;1.005];    % rule for changing rho (line 1: iteration, line 2: value)
zeta               = 1;                % constant for geometric decrease of accuracy/barrier parameter
VV                 = [];               % used for warm-restart in Chambolle-Pock algo

%%%%%%%%%%%% problem parameters
[m,n]   = size(x_true);        % image size
[M,N]   = size(H);             % operator size
Htildet = (H.*(H*ones(N,1)))'; % used to compute the preconditioning matrix

%%%%%%%%%%%% creation of the edge detection operator
% uses the FFT to compute efficiently the convolution
Laplacian  = [0 1 0;1 -4 1;0 1 0];
W          = padarray(Laplacian,[m-3+1,n-3+1]/2,'pre');
W          = padarray(W,[m-3-1,n-3-1]/2,'post');
W          = ones(m,n)-fft2(fftshift(W));                                      % edge detection operator
F          = @(x) reshape(W.*fft2(reshape(x,m,n)),N,1)./sqrt(numel(x));        % convolution operator
F_adj      = @(z) reshape(real(ifft2(W.*reshape(z,m,n))),N,1).*sqrt(numel(z)); % adjoint
F_adj_F_1  = @(x) reshape(real(ifft2(fft2(reshape(x,m,n))./(W.^2))),N,1);      % inverse of F_adj*F

%%%%%%%%%%%% definition of useful functions
smooth     = @(xt)     0.5*sum(abs(F(xt)).^2);           % Smooth regularization  
obj        = @(xt,xg)  smooth(xt)+weightTV*calcTV(xg);   % objective function
TV         = @(x)      calcTV(x);                        % total variation
L          = @(x)      LinOp(x);                         % gradient operator (vertical, horizontal)
Lt         = @(x)      LinOpT(x);                        % transpose of the gradient operator 
prox_TV = @(u,u_ref,lambda,A_1,norm_A_1,VV,A,precision) ...
    prox_TV_metric(u,u_ref,lambda,A_1,norm_A_1,VV,A,precision,L,Lt,N,TV,obj); % proximity operator of TV

%%%%%%%%%%%% find an feasible point to initialize PIPA
[ bool, xg, time_init ] = find_feasible_point( H, y, chi ); % initialization of geometry
xt                      = zeros(N,1);                       % initialization of texture

% constraints value of initial point
[C1,C2,C3,C4,Cfull] = constraints(xt,xg);

if bool==0
    x=[xt;xg];
    return
else
    %%%%%%%%%%%% Start PIPA
    fprintf('\n %%%%%% Start PIPA %%%%%%\n')
    fprintf('iter %d f+g %d \n',0,obj(xt,xg))
    iter = 1;
    time = 0;
    
    while time < time_max
        %%%%%% store variable to plot figures
        obj_vec(iter)      =  obj(xt,xg);                                      % objective function
        snr_vec(iter)      = -20*log10(norm(x_true(:)-xt-xg)/norm(x_true(:))); % signal-to-noise ratio
        x_x_inf_vec(iter)  =  norm(x_inf-[xt;xg])/norm(x_inf);                 % distance to solution
        time_vec(iter)     =  time + time_init;                                % time
        
        if mod(iter,20)==0
            fprintf('iter %d time %.1f obj %.3d\n',iter,time,obj_vec(iter))
        end
        
        tic;
        %%%%% Update algo parameters
        [min_val,index] = min(abs(eps1_rule(1,:)-iter));
        if min_val==0;  eps1= eps1_rule(2,index);  end;
        [min_val,index] = min(abs(rho_rule(1,:)-iter));
        if min_val==0;  rho = rho_rule(2,index);   end;

        %%%%% Check stopping criterion  
        if iter>1 && norm(RX1(:))<eps1*mu/zeta && norm(RX2(:))<eps2*mu/zeta && abs(RX3)<eps3*mu/zeta
            mu   = mu/rho;                      % decrease barrier parameter
            zeta = zeta*1.00001;                % because eps must decrease faster than mu
            precision = max(min(1e-5,mu),1e-7); % precision for computing the prox of TV
        end  

        %%%%% build preconditioning matrix
        temp = mu.*(1./C1-1./C2+H'*(1./C3-1./C4));
        D1   = mu.*(1./(C1.^2)+1./(C2.^2));
        D2   = mu.*(1./(C3.^2)+1./(C4.^2));
        G    = D1 + Htildet*D2;

        A    = @(ut,ug) [F_adj(F(ut))+G.*(ut+ug); G.*(ut+ug)]; % preconditioner
        A_1  = @(x)     [F_adj_F_1(x(1:N)-x(N+1:end)); ...     % inverse of preconditioning operator
                         F_adj_F_1(x(N+1:end)-x(1:N)) + x(N+1:end)./G]; 
                     
        norm_A_1 = CalculNorme(A_1,[xt;xg]);  % compute norm of inverse of preconditioner

        %%%%% store current iterate
        xt_old     = xt;
        xg_old     = xg;
        Cfull_old  = Cfull; % constraints

        %%%%% start backtracking
        gamma = gamma_max;
        for iter1=1:maxiter_backtrack  
             x = [xt_old;xg_old]-gamma.*[xt_old;-xt_old-temp./G];
             
             %%%%% forward-backward iteration
             [xt,xg,VV] = prox_TV(x,[xt_old;xg_old],gamma*weightTV,A_1,norm_A_1,VV,A,precision);
             [C1,C2,C3,C4,Cfull] = constraints(xt,xg);
             
             %%%%% check if feasible point
             if Cfull>0    
                 upper_bound = sum([xt_old-xt;xg_old-xg].*A(xt_old-xt,xg_old-xg))*delta_backtrack/gamma;
                 criterion   = smooth(xt)-smooth(xt_old)-mu*sum(log(Cfull./Cfull_old))+...
                     sum((xt+xg-xt_old-xg_old).*temp)-sum((xt-xt_old).*F_adj(F(xt_old)));
                 
                 %%%%% check if sufficient decrease
                 if criterion <= upper_bound ; break; end 
             end
             
             if iter1==maxiter_backtrack ; disp('Backtracking did not converge'); x=[xt;xg]; return; end
             gamma = gamma*theta_backtrack; % decrease gamma
        end

         %%%%% for stopping criterion
         RX1  = [xt_old-xt;xg_old-xg];
         RX2  =  A(xt_old-xt,xg_old-xg)./gamma;
         RX3  =  sum((Cfull(:)./Cfull_old(:)))-(2*N+2*M);

         time = time + toc;  
         iter = iter + 1;
    end
    x=[xt;xg];
    
    %%% recap
    fprintf('------------------------------------------------------\n')
    fprintf('Duration for initialization: %.1f s\n', time_init)
    fprintf('Duration for %d iterations of PIPA after initialization: %.1f s\n', iter, time)
    fprintf('Final objective function value: %.2f\n', obj_vec(end))
    fprintf('Final normalized distance to solution: %.2f\n', x_x_inf_vec(end))
    fprintf('Final SNR: %.2f\n', snr_vec(end))
    fprintf('------------------------------------------------------\n')
end
 
function tv = calcTV(x)
   X = reshape(x,m,n);                   
   U = X;
   U(:,2:n) = X(:,2:n)-X(:,1:(n-1));    
   V = X;
   V(2:m,:) = X(2:m,:)-X(1:(m-1),:);    
   tv = sum(sqrt(U(:).^2 + V(:).^2));
end

function Dx = LinOp(x)
    X = reshape(x(N+1:end),m,n);
    Du = X;
    Du(:,2:n) = X(:,2:n)-X(:,1:(n-1));   
    Dv = X;
    Dv(2:m,:) = X(2:m,:)-X(1:(m-1),:);   
    Dx = [Du(:);Dv(:)];                     
end

function Dtz = LinOpT(z)
    Z = [reshape(z(1:N),m,n);reshape(z(N+1:end),m,n)];
    Zv = Z(1:m,:);        
    Zh = Z(m+1:2*m,:);   
    U = Zv;                                        
    U(:,1:(n-1)) = Zv(:,1:(n-1))-Zv(:,2:n);    
    V = Zh;                                      
    V(1:(m-1),:) = Zh(1:(m-1),:)-Zh(2:m,:);    
    Dtz = U + V;                                  
    Dtz = [zeros(N,1);Dtz(:)];
end

function [C_1,C_2,C_3,C_4,C_full] = constraints(ut,ug)
    Hutug   =  H*(ut+ug);
    C_1     =  ut+ug;
    C_2     = -ut-ug+1;
    C_3     =  Hutug-y(:)+chi;
    C_4     = -Hutug+y(:)+chi;
    C_full =  [C_1;C_2;C_3;C_4];
end

end




