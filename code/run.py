import time, sys
import inspect
import fire

import tweet_classifier

class Client:

    def tweet_classifier(self, task="evaluate_and_error_analysis"):
        sig = inspect.signature(self.tweet_classifier)
        for param in sig.parameters.values(): print(f'{param.name:15s} = {eval(param.name)}')

        obj = tweet_classifier.TweetClassifier()
        try:
            func = getattr(obj, task)
        except AttributeError:
            print(f"\n- error: method \"{task}\" not found\n")
            sys.exit()

        func()

if __name__ == "__main__":
    tic = time.time()
    fire.Fire(Client)
    print(f'\ntime used: {time.time()-tic:.1f} seconds\n')
