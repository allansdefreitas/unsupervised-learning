"""
III) A MATRIX DE CONFUSÃO DE UMA PARTIÇÃO CRISP VERSUS A OUTRA;
"""
def run_show_confusion_matrix_crisp_vs_crisp(ds_list, m_list, save_fig=True, show_plot=True):


 
            labels_crisp1 = get_labels_from_crisp_or_fuzzy(crisp1)

            #crisp 2
            filename_crisp_2 = ds2 + '_m=' +str(m2) + '_crisp_partition'
            crisp2 = load_matrix_csv(filename=filename_crisp_2)

            #get labels crisp2
            labels_crisp2 = get_labels_from_crisp_or_fuzzy(crisp2)

            #plot the confusion matrix between crisp1 and crisp2
            plot_confusion_matrix(labels_crisp1, labels_crisp2, labels_clusters_names, save_fig=save_fig,
                                  filename=comparison_title, dataset_name=None, plot_title=comparison_title, show_plot=show_plot)

            print("\n")

    print('\n\n')

  print(counter)