MPG123="mpg123-1.28.2"

curl -sL https://downloads.sourceforge.net/sourceforge/mpg123/${MPG123}.tar.bz2 > ${MPG123}.tar.bz2
sha512sum -c mpg123.sha512

tar xzf ${MPG123}.tar.bz2
cd $MPG123
./configure --enable-int-quality --disable-debug
make
make install
cd ..
