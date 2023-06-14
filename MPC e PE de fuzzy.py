#MEDIR MPC E PE

   mpc = calculate_mpc(U)
      pe = calculate_partition_entropy(U)

      crisp = fuzzy_to_crisp_partition(U)

      labels_posteriori = get_labels_from_crisp_or_fuzzy(crisp)

      #put it together with the true labels (labels_a_priori) to calculate the ARI
      ari = calculate_ari(labels_posteriori, labels_a_priori)
      f_measure = calculate_f_measure(labels_posteriori, labels_a_priori)


      print(dataset_name + ' m= ' + str(m) + ' | obj func= %.2f' % round(objective_value,2) + ' |'
      + ' FUZZY MPC=  %.2f' % round(mpc,2) + ' | '
      + ' FUZZY PE= %.2f' % round(pe, 2) + ' | '
      + ' CRISP ARI= %.2f' % round(ari, 2) + ' | '
      + ' CRISP F-measure= %.2f' % round(f_measure, 2) )
