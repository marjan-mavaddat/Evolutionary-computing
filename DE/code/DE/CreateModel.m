function model=CreateModel()
    v=[75    17    22    21    63    71    67    76    45    49 ...
        83    32    88    44    12    70    12    66    40    46 ...
        20    54    87    84    78    41    32    35    72    62 ...
        83    87    49    74    85    63    13    86    74    67 ...
        61    88    74    87    64    23    17    12    25    71];

    w=[486	798	1152	1443	1277	590	592	500	206	281 ...
        1052	1444	457	866	456	375	1263	1160	175	896 ...
        1017	576	808	294	1240	451	919	1155	843	757 ...
        327	919	1079	309	441	963	870	633	1191	116 ...
        266	413	1348	460	1401	763	1384	895	1408	572];

    n=numel(v);
    
    W=10000;
    
    model.n=n;
    model.v=v;
    model.w=w;
    model.W=W;

end