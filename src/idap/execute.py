import augmentation as aug


def GMM_execute(original_data, num_gen, para):
    gmm = aug.Gmm(N=num_gen, n_components=para[0])
    gmm_gen = gmm.fit(original_data)
    return gmm_gen


def GAN_execute(original_data, num_gen, para):
    gan = aug.GAN(num_gen=num_gen, num_epoch=para[0], lr=para[1],
                  batch_size=para[2], latent_dim=para[3])
    list_net, gen_data = gan.fit(original_data)
    return list_net, gen_data


def GNI_execute(original_data, num_gen, para):
    gni = aug.GNI(num_gen=num_gen, mean=para[0], variance=para[1])
    gni_gen = gni.fit(original_data)
    return gni_gen


def KNNMTD_execute(original_data, num_gen, para):
    knnMTD = aug.kNNMTD(n_obs=num_gen, k=para[0])
    knnMTD_gen = knnMTD.fit(original_data)
    return knnMTD_gen


def LLE_execute(original_data, num_gen, para):
    lle = aug.Lle(num_gen=num_gen, n_neighbor=para[0], reg=para[1],
                  n_component=para[2])
    lle_gen = lle.fit(original_data)
    return lle_gen


def MTD_execute(original_data, num_gen, para):
    mtd = aug.MTD(n_obs=num_gen)
    MTD_gen = mtd.fit(original_data)
    return MTD_gen


def SMOTE_execute(original_data, num_gen, para):
    smote = aug.Smote(N=num_gen, k=para[0], r=2)
    smote_gen = smote.fit(original_data)
    return smote_gen
