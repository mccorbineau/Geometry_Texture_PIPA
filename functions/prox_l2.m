function z = prox_l2(x,coeff,N)

%====================================================================
% Kindly report any suggestions or corrections to
% marie-caroline.corbineau@centralesupelec.fr
%
% Input:  x: point at which the prox must be calculated
%         coeff: multiplicative coefficient (stepsize for prox)
%         N: number of pixels
%
% Output: z: argmin 0.5*||zu-xu||^2_2 + 0.5*||zv-xv||^2_2 + 
%                        coeff*\sum_{i=1}^N \sqrt(zu_i^2+zv_i^2)
%
% Computes the proximity operator of the l2 norm.
%====================================================================

u  = x(1:N);
v  = x(N+1:end);
zu = zeros(N,1);     
zv = zu;

sqrtuv           = sqrt(u.^2 + v.^2);
zu(sqrtuv>coeff) = (1-coeff./sqrtuv(sqrtuv>coeff)).*u(sqrtuv>coeff);    
zv(sqrtuv>coeff) = (1-coeff./sqrtuv(sqrtuv>coeff)).*v(sqrtuv>coeff);
z                = [zu;zv];