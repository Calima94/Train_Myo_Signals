import mod_sig_emg as mse
from param_to_use import ParametersToUse


def train_signals_emg(args=None):
    # Define the parameters to use in the training
    param_ = ParametersToUse()

    # Read and pre-processing the data
    new_data = mse.read_data(param_.file)
    # Spare classes from the data
    classes = mse.spare_classes_(data=new_data)
    # Standardize classes to multiple of a window time(ms) / 5
    classes = mse.standardize_classes(classes=classes,
                                      time_between_captures_of_samples=param_.time_between_captures_of_samples,
                                      window_time=param_.window_time)
    # Pass the classes to numpy vector
    classes = mse.to_numpy_func(classes=classes)
    # Sample the classes
    class_mod_ = mse.sample_classes_(classes=classes,
                                     time_between_captures_of_samples=param_.time_between_captures_of_samples,
                                     window_time=param_.window_time,
                                     n_of_channels_and_category=param_.n_of_channels_and_category
                                     )

    class_mod_ = mse.filter_signal(class_mod_=class_mod_,
                                   sos=param_.sos_high_pass_,
                                   n_of_channels_and_category=param_.n_of_channels_and_category)

    class_mod_ = mse.filter_signal(class_mod_=class_mod_,
                                   sos=param_.sos_bandstop_,
                                   n_of_channels_and_category=param_.n_of_channels_and_category)

    # filter the signals with Wavelet filter
    class_mod_ = mse.select_wavelet_layer_x(class_mod_=class_mod_,
                                            filter_to_use=param_.filter_to_use,
                                            levels_to_use=param_.levels_to_use,
                                            layers_to_catch=param_.layers_to_catch,
                                            n_of_channels_and_category=param_.n_of_channels_and_category
                                            )
    # Extract the RMVS or MAV values of the samples
    m_matrix_ = mse.matrix_m(type_matrix=param_.type_matrix,
                             class_mod_=class_mod_,
                             time_between_captures_of_samples=param_.time_between_captures_of_samples,
                             window_time=param_.window_time,
                             n_of_channels_and_category=param_.n_of_channels_and_category
                             )

    # Adjust the M_classes in the final M_matriz data for classifier
    m_matrix_, lens_table_ = mse.fitting_m_class(m_matrix_=m_matrix_)
    df = mse.transf_to_df_class(m_matrix_=m_matrix_, lens_table_=lens_table_)
    df.to_csv("M_Class_Data/training_matrix_csv_m_class.csv", index=False)

    # Split Train Data and Test Data
    [m_class_Train_IndepentVars, m_class_Train_TargetVar, m_class_Test_IndepentVars,
     m_class_Test_TargetVar] = mse.sklearn_spare_test_train(dataframe=df,
                                                            test_size=param_.test_size,
                                                            random_state=param_.random_state)

    # If needed decomposition the PCA componentes of the R matriz to reduce the dimension
    # m_class_Train_IndepentVars_PCA = decomposition_PCA(m_class_Train_IndepentVars, 0.95)

    # Pass the Data to Numpy Vector
    [m_class_Train_IndepentVars, m_class_Train_TargetVar, m_class_Test_IndepentVars,
     m_class_Test_TargetVar] = mse.m_class_to_numpy_(m_class_Train_IndepentVars,
                                                     m_class_Train_TargetVar,
                                                     m_class_Test_IndepentVars,
                                                     m_class_Test_TargetVar
                                                     )

    # Encode the classes
    [m_class_Train_TargetVar, m_class_Test_TargetVar] = mse.encode_data(m_class_Train_TargetVar=m_class_Train_TargetVar,
                                                                        m_class_Test_TargetVar=m_class_Test_TargetVar
                                                                        )
    # Save the preditors and calc the erros of each preditor
    [error_tree_n, error_lda_n, error_gnb_n, error_lin_svm_n, error_neigh_n] = mse.apply_classifiers(
        var_train=m_class_Train_IndepentVars,
        var_target=m_class_Train_TargetVar.ravel(),
        var_test=m_class_Test_IndepentVars,
        var_test_target=m_class_Test_TargetVar.ravel(),
        cv=param_.cv)


#if __name__ == "__main__":
#    main()
