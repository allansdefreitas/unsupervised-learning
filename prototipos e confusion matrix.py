"""
PROTÓTIPOS

"""
def run_centroids_confusion_matrix_vs_a_priori(save_fig=False, show_plot=False):

  print("PROTÓTIPOS ######################################")

  for i in range(NUMBER_OF_DATASETS):

    mpc = calculate_mpc(U)
    pe = calculate_partition_entropy(U)

    crisp = fuzzy_to_crisp_partition(U)

    labels_posteriori = get_labels_from_crisp_or_fuzzy(crisp)

    dataset_feature_labels = datasets_features_labels[i]

    title_centroids = dataset_name  + '_m=' + str(m) + ' Centroids'
    print(title_centroids)
    plot_centroids_as_table(centroids, dataset_feature_labels, save_fig=save_fig, filename=title_centroids,
                            dataset_name=dataset_name, plot_title=title_centroids, show_plot=show_plot)
    #print("\n\n")

    title_cm = "Confusion matrix: " + dataset_name + '_m=' + str(m) + ' vs a priori'
    print(title_cm)
    plot_confusion_matrix(labels_a_priori, labels_posteriori, labels_clusters_names, save_fig=save_fig,
                          filename=title_cm, dataset_name=dataset_name, plot_title=title_cm, show_plot=show_plot)
    #print("\n\n")

    print("\n")