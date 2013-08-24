#!/bin/bash
#requires ImageMagick to convert

#output - 400px of letter

pathSourceLetters=/letters	#where letters in jpg are exists
destinationFolder=data		#output folder

if [ ! -d "$destinationFolder" ]; then
	mkdir -p $destinationFolder ;	
fi;

iterator=0
for letter in "$pathSourceLetters"/* ; do
		temp=$(convert $letter -trim -resize 20x20! -monochrome -compress None pbm:- | tail -n +3)
		inOutData=
		
		for word in $temp ; do
			inOutData="$inOutData$word,"		
		done
		echo $letter

		echo ${inOutData%?} > "$destinationFolder/$iterator".txt
		let iterator+=1		
done



