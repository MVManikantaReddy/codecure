
#include <stdio.h>
int QuickSort(int arr[], int start, int end)
{
    int Pivot = arr[end];
    int i = start - 1, j = 0;
    for (j = start; j < end; j++)
    {
        if (arr[j] < Pivot)
        {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    int temp = arr[i + 1];
    arr[i + 1] = arr[end];
    arr[end] = temp;
    return i + 1;
}

void Partition(int arr[], int start, int end)
{
    if (start < end)
    {
        int Pivot = QuickSort(arr, start, end);
        Partition(arr, start, Pivot - 1);
        Partition(arr, Pivot + 1, end);
    }
}

int main()
{
    int n, i;
    printf("Enter No.of Elements : ");
    scanf("%d", &n);
    int arr[n];
    for (i = 0; i < n; i++)
    {
        printf("Enter Element-%d: ", i + 1);
        scanf("%d", &arr[i]);
    }
    printf("Before Merge Sort\n");
    printf("=================\n");
    for (i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n\n");
    Partition(arr, 0, n - 1);

    printf("After Merge Sort\n");
    printf("=================\n");
    for (i = 0; i < n; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}