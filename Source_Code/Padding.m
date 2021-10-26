function X=Padding(Img,k)


[r,c]=size(Img);
rows=r + (2*k) ;
cols= c + (2*k);
X=NaN(rows ,cols);


for i=1:r
    for j=1:c
        X(i+k,j+k)=Img(i,j);
    end
end

