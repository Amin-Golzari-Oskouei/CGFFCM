function V_ij=VijCalculator(HSVchannel,d,k,mu_ij)

PadImg=Padding(HSVchannel,k);
Mu_ij=Padding(mu_ij,k);

[ro,co,~]=size(PadImg); %% Size of Padding Image
PadV_ij=NaN(ro,co);

%% Calculates V_ij
for i=1+k : ro-k
    for j=1+k : co-k
        sigma=0;
        sum=0;
        absNp=0;
        for p= i-((d-1)/2) : i+((d-1)/2) %% Location Of Window d*d
            for q=j-((d-1)/2) : j+((d-1)/2)
                if  (isnan(PadImg(p,q))) ==0
                    absNp=absNp+1;   %% Number Of non-NaN Neighbors in PadH centered at (p,q)
                    sigma=sum + (PadImg(p,q)- Mu_ij(i,j))^ 2;
                    sum = sigma;
                end
            end
        end
        PadV_ij(i,j)= sqrt(sigma /(absNp)) ;
    end
end

v_ij=zeros(ro - 2*k,co - 2*k);
for i=1+k : ro-k
    for j=1+k : co-k
        v_ij(i-k,j-k)=PadV_ij(i,j);
    end
end

v_max=max(max(v_ij));
V_ij=v_ij / v_max;

