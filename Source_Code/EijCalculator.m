function E_ij=EijCalculator(imGray)

[dx,dy] = gradient(imGray);
e_ij=sqrt(dx.^2 + dy.^2);
e_max=max(max(e_ij));
E_ij=e_ij / e_max;


