function z = fNMI(class, lable)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length
% Ouput:
%   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(class) == numel(lable));
n = numel(class);
class = reshape(class,1,n);
lable = reshape(lable,1,n);

l = min(min(class),min(lable));
class = class-l+1;
lable = lable-l+1;
k = max(max(class),max(lable));

idx = 1:n;
Mx = sparse(idx,class,1,n,k,n);
My = sparse(idx,lable,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));


% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
z = max(0,z);

