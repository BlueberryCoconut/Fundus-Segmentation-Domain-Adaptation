class MYPath(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'fundus':
            # 🌟 建议直接换成你电脑上的绝对路径
            return r'C:\Users\18268\Desktop\Fundus'
        if database =='cell':
            return r'C:\Users\18268\Desktop\data'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
