#include <stdio.h>
#include <stdlib.h>


int main() {
  FILE *fp;
  int i=0;
  fp = fopen("test103", "rb");
  char buffer[10];  

unsigned short stringLength = 0;
//fread(&stringLength, sizeof(unsigned short), 1, fp);

//char *drumReadString = malloc(sizeof(char) * stringLength);
fread(&buffer, 1, 10, fp);
//int count = fread(drumReadString, sizeof(char), stringLength, fp);

printf("%s\n", buffer);

fclose(fp); 

  return 0;
}