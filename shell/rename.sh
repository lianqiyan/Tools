cnt=0
IFS=$(echo -en "\n\b")
echo -en $IFS
new_name='train_folder'
for folder in *.mp4
do
    new_folder=$new_name$cnt'.mp4'
    mv $folder $new_folder
    cnt=`expr $cnt + 1`
    echo $new_folder
done
