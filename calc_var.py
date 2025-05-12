import pickle
import numpy as np
import matplotlib.pyplot as plt
## Get files with prefix attn_save under the folder
import os
import glob
from scipy import stats

def get_mask_attn_pkl_files(folder_path,prefix):
    pattern = os.path.join(folder_path,prefix+'*.pkl')
    return glob.glob(pattern)

def element_wise_variance(arr1, arr2):
    # Ensure that the input arrays have the same length
    assert len(arr1) == len(arr2), "Arrays must have the same length"

    # Calculate the variance of the corresponding positions
    variances = np.zeros(len(arr1))
    for i in range(len(arr1)):
        # There are only two values for each position, so use a simple variance formula
        mean = (arr1[i] + arr2[i]) / 2
        variances[i] = ((arr1[i] - mean)**2 + (arr2[i] - mean)**2) / 2

    return variances

def calculate_statistics(data1, data2):
    """Calculate statistical features of two groups of data, including t-test, confidence interval and effect size"""
    # t-test
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    
    # Effect size - Cohen's d
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) / 
                         (len(data1) + len(data2) - 2))
    cohens_d = (mean2 - mean1) / pooled_std
    
    # Calculate 95% confidence interval
    ci_low, ci_high = stats.t.interval(0.95, len(data1)+len(data2)-2, 
                                      loc=mean2-mean1, 
                                      scale=stats.sem(np.concatenate([data1, data2])))
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_diff": mean2 - mean1,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high
    }

# Please replace the path here with your actual folder path
folder_path = './'
mask_files = get_mask_attn_pkl_files(folder_path,'mask_attn')
#print(files)
org_files = get_mask_attn_pkl_files(folder_path,'org_attn')
mask_files.sort()
org_files.sort()
print(mask_files)
print(org_files)
with open('./position.pkl', 'rb') as f:
    postions = pickle.load(f)
print(postions)
len_mask=len(mask_files)
len_org=len(org_files)
read_len=min(len_mask,len_org)
step=0
a=0
mask_target=None
org_target=None
count=0
mask_var_sum=0
org_var_sum=0
diff_var_sum=0
diff_var_sum_list=[]
mask_mean_list=[]
org_mean_list=[]
mask_mean_sum=0
org_mean_sum=0
diff_mean_sum=0
diff_mean_sum_list=[]
var_flag=False
var_precent_sum=0
nomask_postion=[i for i in range(1024)]
nomask_postion=[i for i in nomask_postion if i not in postions]
print(nomask_postion)

# Used to store statistical analysis results
all_mask_means = []
all_org_means = []

if var_flag:
    with open('./var_mean_diff_new_0318.txt','a') as var_diff_file:
        for mask_file,org_file in zip(mask_files[:read_len],org_files[:read_len]):
            count+=1
            print(count)
            with open(mask_file, 'rb') as f:
                mask_data = pickle.load(f)
            with open(org_file, 'rb') as f:
                org_data = pickle.load(f)

            mask_mean=np.mean(mask_data, axis=1)[0,-1,:][nomask_postion]
            org_mean=np.mean(org_data, axis=1)[0,-1,:][nomask_postion]
            
            # Store the mean of all samples for subsequent statistical analysis
            all_mask_means.append(mask_mean)
            all_org_means.append(org_mean)
            
            mask_var_sum+=np.var(mask_mean)
            org_var_sum+=np.var(org_mean)
            diff_var_sum+=np.var(org_mean)-np.var(mask_mean)
        
        print(f'up % = mask-org/mask = {(mask_var_sum-org_var_sum)/mask_var_sum}')
        
        # Perform statistical analysis
        if len(all_mask_means) > 1 and len(all_org_means) > 1:
            # Convert list to numpy array for analysis
            mask_means_array = np.array(all_mask_means).flatten()
            org_means_array = np.array(all_org_means).flatten()
            
            # Calculate statistical features
            stats_results = calculate_statistics(mask_means_array, org_means_array)
            
            print("\n===== Statistical Analysis Results =====")
            print(f"t-statistic: {stats_results['t_statistic']:.4f}")
            print(f"p-value: {stats_results['p_value']:.6f}")
            print(f"Cohen's d effect size: {stats_results['cohens_d']:.4f}")
            print(f"Mean difference (org - mask): {stats_results['mean_diff']:.6f}")
            print(f"95% confidence interval: [{stats_results['ci_95_low']:.6f}, {stats_results['ci_95_high']:.6f}]")
            
            # Write results to file
            var_diff_file.write("\n===== Statistical Analysis Results =====\n")
            var_diff_file.write(f"Sample size: {count}\n")
            var_diff_file.write(f"t-statistic: {stats_results['t_statistic']:.4f}\n")
            var_diff_file.write(f"p-value: {stats_results['p_value']:.6f}\n")
            var_diff_file.write(f"Cohen's d effect size: {stats_results['cohens_d']:.4f}\n")
            var_diff_file.write(f"Mean difference (org - mask): {stats_results['mean_diff']:.6f}\n")
            var_diff_file.write(f"95% confidence interval: [{stats_results['ci_95_low']:.6f}, {stats_results['ci_95_high']:.6f}]\n")
else:
    with open('./seq_mean_diff_new_0318.txt','a') as var_diff_file:
        for mask_file,org_file in zip(mask_files[:read_len],org_files[:read_len]):
            count+=1
            print(count)
            with open(mask_file, 'rb') as f:
                mask_data = pickle.load(f)
            with open(org_file, 'rb') as f:
                org_data = pickle.load(f)
            mask_mean=np.mean(mask_data, axis=1)[0,-1,:][postions]
            org_mean=np.mean(org_data, axis=1)[0,-1,:][postions]
            
            # Store the mean of all samples for subsequent statistical analysis
            all_mask_means.append(mask_mean)
            all_org_means.append(org_mean)
            
            diff_mean_sum+=np.mean(org_mean)-np.mean(mask_mean)
            mask_mean_sum+=np.mean(mask_mean)
            org_mean_sum+=np.mean(org_mean)
            
        print(f'mask seq mean:{mask_mean_sum/count}')
        print(f'org seq mean:{org_mean_sum/count}')
        print(f'org-mask = mean diff:{diff_mean_sum/count}')
        print(f'down % = org-mask/org = {(org_mean_sum-mask_mean_sum)/org_mean_sum}')  
        print(count)
        
        # Perform statistical analysis
        if len(all_mask_means) > 1 and len(all_org_means) > 1:
            # Convert list to numpy array for analysis
            mask_means_array = np.array(all_mask_means).flatten()
            org_means_array = np.array(all_org_means).flatten()
            
            # Calculate statistical features
            stats_results = calculate_statistics(mask_means_array, org_means_array)
            
            print("\n===== Statistical Analysis Results =====")
            print(f"t-statistic: {stats_results['t_statistic']:.4f}")
            print(f"p-value: {stats_results['p_value']:.6f}")
            print(f"Cohen's d effect size: {stats_results['cohens_d']:.4f}")
            print(f"Mean difference (org - mask): {stats_results['mean_diff']:.6f}")
            print(f"95% confidence interval: [{stats_results['ci_95_low']:.6f}, {stats_results['ci_95_high']:.6f}]")
            
            # Write results to file
            var_diff_file.write("\n===== Statistical Analysis Results =====\n")
            var_diff_file.write(f"Sample size: {count}\n")
            var_diff_file.write(f"t-statistic: {stats_results['t_statistic']:.4f}\n")
            var_diff_file.write(f"p-value: {stats_results['p_value']:.6f}\n")
            var_diff_file.write(f"Cohen's d effect size: {stats_results['cohens_d']:.4f}\n")
            var_diff_file.write(f"Mean difference (org - mask): {stats_results['mean_diff']:.6f}\n")
            var_diff_file.write(f"95% confidence interval: [{stats_results['ci_95_low']:.6f}, {stats_results['ci_95_high']:.6f}]\n")
