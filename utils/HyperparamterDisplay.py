from __future__ import print_function, absolute_import


def display(args):
    #  Display information of current training
    print('Learn Rate  \t%.1e' % args.lr)
    print('Epochs  \t%05d' % args.epochs)
    print('Log Path \t%s' % args.save_dir)
    print('Network \t %s' % args.net)
    print('Data Set \t %s' % args.data)
    print('Batch Size  \t %d' % args.batch_size)
    print('Num-Instance  \t %d' % args.num_instances)
    print('Embedded Dimension \t %d' % args.dim)
    print('freeze_BN \t', args.freeze_BN)
    print('Loss Function \t%s' % args.loss)
    # print('Number of Neighbour \t%d' % args.k)
    print('Alpha \t %f' % args.alpha)
    print('Beta \t %f' % args.beta)
    print('Begin to fine tune %s Network' % args.net)
    print(40*'#')
