function mu_ij=MuCalculator(HSVchannel,d,k)

PadImg=Padding(HSVchannel,k);

[ro,co,~]=size(PadImg); %% Size of Padding Image
Padmu_ij=NaN(ro,co);

%% Calculates mu_ij
for i=1+k : ro-k
    for j=1+k : co-k
        sigma=0;
        sum=0;
        absNp=0;
        for p= i-((d-1)/2) : i+((d-1)/2) %% Location Of Window d*d
            for q=j-((d-1)/2) : j+((d-1)/2)
                if  (isnan(PadImg(p,q))) ==0
                    absNp=absNp+1;   %% Number Of non-NaN Neighbors in PadH centered at (p,q)
                    sigma=sum + PadImg(p,q);
                    sum = sigma;
                end
            end
        end
        Padmu_ij(i,j)= sigma /(absNp) ;
    end
end

mu_ij=zeros(ro - 2*k,co - 2*k);
for i=1+k : ro-k
    for j=1+k : co-k
        mu_ij(i-k,j-k)=Padmu_ij(i,j);
    end
end




