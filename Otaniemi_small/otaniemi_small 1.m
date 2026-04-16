  try
      viewer = siteviewer(SceneModel="small_otaniemi.glb",ShowOrigin=false);
  catch ME
      disp(ME.message)
      disp(ME.stack(1))
  end

%array = phased.ULA('NumElements',8,'Element',phased.IsotropicAntennaElement,'ElementSpacing',0.04);
tx1 = txsite("cartesian",AntennaPosition=[-5; 70; 19],TransmitterFrequency=3.6e9,Name="a"); %,AntennaAngle=45,Antenna = array);
tx2 = txsite("cartesian",AntennaPosition=[-33; -17; 19],TransmitterFrequency=3.6e9,Name="b"); %,AntennaAngle=90,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx3 = txsite("cartesian",AntennaPosition=[-3; 14; 19],TransmitterFrequency=3.6e9,Name="c"); %,AntennaAngle=45,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx4 = txsite("cartesian",AntennaPosition=[-6; -40; 19],TransmitterFrequency=3.6e9,Name="d"); %,AntennaAngle=0,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx5 = txsite("cartesian",AntennaPosition=[55; -52.5; 19],TransmitterFrequency=3.6e9,Name="e"); %,AntennaAngle=0,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx6 = txsite("cartesian",AntennaPosition=[61; 7; 19],TransmitterFrequency=3.6e9,Name="f"); %,AntennaAngle=90,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx7 = txsite("cartesian",AntennaPosition=[49; 46; 19],TransmitterFrequency=3.6e9,Name="g"); %,AntennaAngle=-45,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx8 = txsite("cartesian",AntennaPosition=[28; -41; 19],TransmitterFrequency=3.6e9,Name="h"); %,AntennaAngle=0,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
tx9 = txsite("cartesian",AntennaPosition=[84; -18; 19],TransmitterFrequency=3.6e9,Name="i"); %,AntennaAngle=90,Antenna = arrayConfig('Size',[8 1],'ElementSpacing',0.04));
show(tx1); show(tx2);show(tx3);show(tx4);show(tx5);show(tx6);show(tx7);show(tx8); show(tx9);
sites={tx1,tx2,tx3,tx4,tx5,tx6,tx7,tx8,tx9};

pm = propagationModel("raytracing",CoordinateSystem="cartesian",SurfaceMaterial="auto");
pm.MaxNumDiffractions=1;
pm.MaxNumReflections=4;

%rxx=[-45:5:85]; rxy=[-60:5:85]; zf=1.25; id='first';
rxx=[-47.5:5:85]; rxy=[-60:5:85]; zf=1.25; id='second';
%rxx=[-45:5:85]; rxy=[-62.5:5:85]; zf=1.25; id='third';
%rxx=[80:5:85]; rxy=[-60:5:85]; zf=1.25; id='zero';

for k=1:numel(sites)
  txs=sites{k};
  ccount=0;
  for x = rxx
    for y = rxy
          
          rx = rxsite("cartesian",AntennaPosition=[x;y;zf]);
          rays=raytrace(txs,rx,pm);
          %disp(rays{1}(1))
          jsonstr=jsonencode(rays);
          filename=strcat('rays/otaniemi_',txs.Name,'_',id,num2str(ccount,'%03d'),'.json'); 
          fid=fopen(filename,'w');
          fprintf(fid,'%s',jsonstr);
          fclose(fid);
          ccount=ccount+1;
      
    end
  end
end