

##CRISP DUAS A DUAS RAND E F-1

# dataset_1 e dataset_2 -----------------------------------------------

partition_a = 'dataset_1'
partition_b = 'dataset_2'


#comparando Rand Index
metric_name = 'Rand Index'

print('Comparing ', partition_a ,  ' e ', partition_b, "--------\n")

ari_a =  rand_index_f_measures[0][0]
ari_b =  rand_index_f_measures[1][0]

print(metric_name + ': ', str(ari_a) +' e ' + str(ari_b))

compare_metrics_higher_better(ari_a, ari_b, partition_a, partition_b, metric_name)


#comparando F-measure
metric_name = 'F-measure'
f1_a =  rand_index_f_measures[0][1]
f1_b =  rand_index_f_measures[1][1]

print(metric_name + ': ', str(f1_a) + ' e ' + str(f1_b) )
compare_metrics_higher_better(f1_a, f1_b, partition_a, partition_b, metric_name)

print("\n\n")

# dataset_1 e dataset_3 -----------------------------------------------
partition_a = 'dataset_1'
partition_b = 'dataset_3'

#comparando Rand Index
metric_name = 'Rand Index'

print('Comparing ', partition_a ,  ' e ', partition_b, "--------\n")

ari_a =  rand_index_f_measures[0][0]
ari_b =  rand_index_f_measures[2][0]

print(metric_name + ': ', str(ari_a) +' e ' + str(ari_b))

compare_metrics_higher_better(ari_a, ari_b, partition_a, partition_b, metric_name)


#comparando F-measure
metric_name = 'F-measure'
f1_a =  rand_index_f_measures[0][1]
f1_b =  rand_index_f_measures[2][1]

print(metric_name + ': ', str(f1_a) + ' e ' + str(f1_b) )
compare_metrics_higher_better(f1_a, f1_b, partition_a, partition_b, metric_name)

print("\n\n")
# dataset_2 e dataset_3 -----------------------------------------------
partition_a = 'dataset_2'
partition_b = 'dataset_3'

#comparando Rand Index
metric_name = 'Rand Index'
print('Comparing ', partition_a ,  ' e ', partition_b, "--------\n")

ari_a =  rand_index_f_measures[1][0]
ari_b =  rand_index_f_measures[2][0]

print(metric_name + ': ', str(ari_a) +' e ' + str(ari_b))

compare_metrics_higher_better(ari_a, ari_b, partition_a, partition_b, metric_name)

#comparando F-measure
metric_name = 'F-measure'
f1_a =  rand_index_f_measures[1][1]
f1_b =  rand_index_f_measures[2][1]

print(metric_name + ': ', str(f1_a) + ' e ' + str(f1_b) )
compare_metrics_higher_better(f1_a, f1_b, partition_a, partition_b, metric_name)