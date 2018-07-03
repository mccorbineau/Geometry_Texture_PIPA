function [ xt,xg,VV ] = prox_TV_metric(u,u_ref,lambda,A_1,norm_A_1,VV,A,precision,L,Lt,N,TV,obj)
%====================================================================
% Kindly report any suggestions or corrections to
% marie-caroline.corbineau@centralesupelec.fr
%
% Input:  u: point at which the prox must be computed
%         u_ref: previous iteration in the main agorithm
%         lambda: coefficient to compute the prox
%         A_1: inverse of the preconditioning matrix
%         norm_A_1: norm of the inverse of the preconditioning matrix
%         VV: warm restart for the dual variable
%         A: preconditioner (used for stopping criterion)
%         precision: precision for computing the prox
%         L,Lt: operator and its adjoint for the image gradient (horizontal and vertical)
%         N: number of pixels
%         TV: function which computes the total variation (used for stopping criterion)
%         obj: objective function for the main problem (used for stopping criterion)
%
% Output: xt: texture
%         xg: geometry
%         VV: warm restart for dual variable
%
% Solves P_I:    minimize      0.5*||[xt;xg]-u||^2_A + lambda*TV(xg)  
%                for (xt and xg in R^N)
%
% Implementation of algorithm 2 in
% A. Chambolle and T. Pock. A first-order primal-dual algorithm for convex
% problems with applications to imaging. Journal of Mathematical Imaging and Vision, 
% Vol. 40, No. 1, pp 120-145, 2011.
%====================================================================


if (isempty(VV))
    VV    = zeros(2*N,1);
    N_min = 100; % minimal number of iteration
else
    N_min=7; % minimal number of iteration
end
N_max = 1000; % maximal number of iteration

%%%%% algorithm parameters
tau   = (1e3/sqrt(8*norm_A_1));
gamma =  0.99;
sigma =  gamma/(tau*8*norm_A_1);

TV_ref    = TV(u_ref(N+1:end));
obj_ref   = obj(u_ref(1:N),u_ref(N+1:end));

% initialization
ut   = u(1:N);
ug   = u(N+1:end);
xbar = u;
x    = u;
xt   = x(1:N);
xg   = x(N+1:end);

crit = lambda*TV(xg)+0.5*sum((x-u).*A(xt-ut,xg-ug)); % objectivefunction for the prox
    
for i = 1:N_max
    crit_old = crit;
    
    temp     = VV+sigma*L(xbar);
    VV       = temp-sigma*prox_l2(temp./sigma,lambda/sigma,N);
    xold     = x;
    x        = (tau.*u+x-tau.*A_1(Lt(VV)))./(1+tau);
    theta    = 1/sqrt(1+2*gamma*tau);
    tau      = theta*tau;
    sigma    = sigma/theta;
    xbar     = x+theta.*(x-xold);  
    
    xt       = x(1:N);
    xg       = x(N+1:end);
    crit     = lambda*TV(xg)+0.5*sum((x-u).*A(xt-ut,xg-ug));

    % stopping criterion
    if i>N_min   
        residu=(crit-crit_old)/crit;
        if obj(xt,xg) < obj_ref && abs(residu) < precision;break;end
        if lambda*TV_ref >= lambda*TV(xg)+sum(A(ut-xt,ug-xg).*(u_ref-x));break;end
    end
    if i==N_max;disp('prox did not converge');end
end  

end


