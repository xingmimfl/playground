#include <ftw.h>
#include <stdio.h>
#include <sys/stat.h>
#include <string.h>

//https://stackoverflow.com/questions/67273/how-do-you-iterate-through-every-file-directory-recursively-in-standard-c 
int list(const char *name, const struct stat *status, int type)
{
     if (type == FTW_NS)
     {
         return 0;
     }

     if (type == FTW_F)
     {
         printf("0%3o\t%s\n", status->st_mode&0777, name);
     }

     if (type == FTW_D && strcmp(".", name) != 0)
     {
         printf("0%3o\t%s/\n", status->st_mode&0777, name);
     }
     return 0;
}

int main(int argc, char *argv[])
{
     if(argc == 1)
     {
         ftw(".", list, 1);
     }
     else
     {
         ftw(argv[1], list, 1);
     }

     return 0;
}
