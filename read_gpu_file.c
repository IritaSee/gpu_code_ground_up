#include <stdio.h>
#include <stdlib.h>


int main() {
  FILE *fp;
  int i=0;
  fp = fopen("core_note", "rb");
  int buffer[1600000];  
  
unsigned short stringLength = 0;
//fread(&stringLength, sizeof(unsigned short), 1, fp);

//char *drumReadString = malloc(sizeof(char) * stringLength);
fread(&buffer, 4, 1600000, fp);
//int count = fread(drumReadString, sizeof(char), stringLength, fp);
for(i=1;i<=1600000;i++){
if(buffer[i-1] != 0 ){
printf("%d, ", buffer[i-1]);
}
if(i%10==0){printf("\n\n");}
}


fclose(fp); 

  return 0;
}