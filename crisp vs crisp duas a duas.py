"""
III) A MATRIX DE CONFUSÃO DE UMA PARTIÇÃO CRISP VERSUS A OUTRA;
"""
def run_confusion_matrix_crisp_vs_crisp(ds_list, m_list, save_fig=True, show_plot=True):


  print("III) A MATRIX DE CONFUSÃO DE UMA PARTIÇÃO CRISP VERSUS A OUTRA; ######################################")
  print("\n\n ")

  counter = 0

  for i_ds in range(len(ds_list)):
    for j_m in range(len(m_list)):

        for k_ds in range(len(ds_list)):
          for l_m in range(len(m_list)):

            ds1 = datasets_names[i_ds]
            ds2 = datasets_names[i_ds]

            m1 = m_list[j_m]
            m2 = m_list[l_m]

            if ds1 == ds2 and m1 == m2:
              continue
            else:
              counter +=1
              part_a = ds1 + ' m=' + str(m1)
              part_b = ds2 + ' m=' + str(m2)
              comparison_title = part_a + ' VS ' + part_b
              print(comparison_title, "\n")

            #print confusion matrix
            #get crisp1 and crisp2 from files
            #crisp 1
            filename_crisp_1 = ds1 + '_m=' +str(m1) + '_crisp_partition'
            crisp1 = load_matrix_csv(filename=filename_crisp_1)

            #get labels crisp1
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