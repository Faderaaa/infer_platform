from celery import Celery


def make_celery(app):
    celery = Celery(  # 实例化Celery
        'tasks',
        broker='redis://localhost:6379/1',  # 使用redis为中间人
        # backend='redis://localhost:6379/2'  # 结果存储
    )

    class ContextTask(celery.Task):  # 创建ContextTask类并继承Celery.Task子类
        def __call__(self, *args, **kwargs):
            with app.app_context():  # 和Flask中的app建立关系
                return self.run(*args, **kwargs)  # 返回任务

    celery.Task = ContextTask  # 异步任务实例化ContextTask
    return celery  # 返回celery对象
