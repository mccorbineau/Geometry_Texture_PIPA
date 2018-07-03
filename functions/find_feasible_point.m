function [ bool, init, time_init ] = find_feasible_point( H, y, chi )

%====================================================================
% Kindly report any suggestions or corrections to
% marie-caroline.corbineau@centralesupelec.fr
%
% Input:  H: observation operator (Radon transform)
%         y: observed data
%         chi: measurement uncertainty
%
% Output: bool: 1 if a feasible point was found, 0 else
%         init: initial point for PIPA
%         time_init: time needed for intialization
%
% Solves P_I:                 minimize      s  for (s in R, x in R^{N})
%                             subject to   -chi-s <= H*x-y <= chi+s
%                                           0 <= x <= 1
%                                           s >= 0
%
% which can be re-written as: minimize    c'*x 
%                             subject to  A*x <= b
%====================================================================

eps   = 1e-8;     % precision for finding a feasible point
[M,N] = size(H);

A  =  [ -ones(M,1)            H;...
        -ones(M,1)           -H;...
         sparse(N,1)     speye(N);...
         sparse(N,1)    -speye(N);...
        -1               sparse(1,N)];
b  =  [y+chi;-y+chi;ones(N,1);sparse(N,1);0];
c  =  [1; sparse(N,1)];

% xc is a strictly feasible point for problem P_I 
xg  =  rand(N,1);
s1  =  max(max(H*xg-y)-chi+1,0);
s2  =  max(max(y-H*xg)-chi+1,0);
xc  =  [max(s1,s2);xg];

% Barrier method starting at xc.
mu = 10; % initialization for the barrier parameter
fprintf('\n %%%%%% Start interior point algo to solve P_I for initialization %%%%%%\n')
tic
x          = lp(A,b,c,xc,mu,1e-4,H,y,chi);
time_init  = toc;
init       = x(2:end);

% check if the point found with IPM is feasible
if min(0,min(init)-eps)==0 &&...
        max(1,max(init)+eps)==1 &&...
        max(chi,abs(min(H*init-y))+eps)==chi &&...
        max(chi,max(H*init-y)+eps)==chi
    bool=1;
    fprintf('\n Found a feasible initial point for PIPA\n')
else
    bool=0;
    fprintf('\n Did not find a feasible point\n')
end

end

