#!/bin/bash
#requires ImageMagick to convert

#output - 400px + 64 fields - each neurons for one letter

pathSourceFonts=/fonts	#where fonts are exists
destinationFolder=data	#output folder
dataName="data".csv		#output file

if [ ! -d "$destinationFolder" ]; then
	mkdir -p $destinationFolder ;	
fi;

iterator=0
for fontName in "$pathSourceFonts"/* ; do
	for letter in {A..Z} {a..z} {0..9} . '\@' ; do		
		temp=$(convert -font "$fontName" -pointsize 72 label:"$letter" -trim -resize 20x20! -monochrome -compress None pbm:- | tail -n +3)
		#convert -font "$fontName" -pointsize 72 label:"$letter" -trim -resize 20x20! -monochrome "$destinationFolder/$iterator".jpg
		inOutData=	
		
		for word in $temp ; do
			inOutData="$inOutData$word,"		
		done		
		
		iter=$(($iterator % 64))
		Out=	
		
		for ((i=0; $i<64; i++)); do
			if [ $i = $iter ]; then
				Out="1,"$Out				
			else
				Out="0,$Out"
			fi;
		done
		
		let iterator+=1
		echo -e "$inOutData${Out%?}" >> "$destinationFolder/$dataName"
		echo "$iterator. Dodano znak $letter (${Binary%?}) z czionki $fontName"
	done
done



